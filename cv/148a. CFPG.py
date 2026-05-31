import torch
import torch.nn as nn
import torch.nn.functional as F


class GRN(nn.Module):
    """
    全局响应归一化（Global Response Normalization）
    作用：
    1. 对空间维度上的整体响应进行归一化；
    2. 增强强响应区域与弱响应区域之间的对比；
    3. 常用于提升卷积块的训练稳定性与表达能力。

    输入:
        x: (B, C, H, W)

    输出:
        与输入同形状的特征图
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 在空间维度上做 L2 范数统计，得到每个通道的全局响应强度
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)

        # 归一化后的响应
        nx = x / (gx + self.eps)

        # 残差式调制
        return x + self.gamma * nx + self.beta


class ChannelReweight(nn.Module):
    """
    通道重标定模块
    作用：
    1. 根据输入特征的全局统计，为每个通道生成一个权重；
    2. 让不同通道对后续的多尺度融合具有不同偏好；
    3. 类似轻量版 SE，但更适合这里的门控场景。
    """
    def __init__(self, dim: int, hidden_ratio: float = 0.25):
        super().__init__()
        hidden_dim = max(8, int(dim * hidden_ratio))

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 生成通道权重，范围在 0~1
        w = self.mlp(self.pool(x))
        return x * w


class CFPG(nn.Module):
    """
    CFPG: Confidence-Feedback Peripheral Gating
    中文名：置信反馈式外围门控模块

    设计目标：
    在原始 PFG / PFGA 思路基础上，进一步引入：
    1. 粗门控（由频率线索生成）
    2. 反馈细化门控（由多尺度分支真实响应反向修正）
    3. 不确定性估计（根据粗门控熵，自适应融合粗门控与细化门控）
    4. 通道重标定（增强不同通道对尺度响应的选择性）

    相比原始版本更像一个“CVPR 风格”的增强块：
    - 有明确的 coarse-to-refine 思路
    - 有 confidence / uncertainty 建模
    - 有反馈机制
    - 有尺度竞争和通道重标定
    """

    class Branch(nn.Module):
        """
        单个多尺度外围分支

        结构说明：
        1. 使用 DW(1xK) + DW(Kx1) 近似 KxK 大卷积核；
        2. 可选中心抑制路径，用于构建“外围 - 中心”对抗结构；
        3. 中心抑制系数是可学习的，并通过 tanh 约束到合理范围。
        """
        def __init__(self, dim: int, k: int, center_suppress: bool = True):
            super().__init__()
            self.k = k
            self.center_suppress = center_suppress

            # 先做水平方向的大核深度卷积
            self.dw_h = nn.Conv2d(
                dim, dim,
                kernel_size=(1, k),
                padding=(0, k // 2),
                groups=dim,
                bias=False
            )

            # 再做垂直方向的大核深度卷积
            self.dw_v = nn.Conv2d(
                dim, dim,
                kernel_size=(k, 1),
                padding=(k // 2, 0),
                groups=dim,
                bias=False
            )

            # 可选：中心 3x3 路径，用于中心抑制
            if self.center_suppress:
                self.dw_c = nn.Conv2d(
                    dim, dim,
                    kernel_size=3,
                    padding=1,
                    groups=dim,
                    bias=False
                )
                self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
            else:
                self.dw_c = None
                self.register_parameter("beta", None)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # 外围大感受野响应
            y = self.dw_v(self.dw_h(x))

            # 中心抑制：外围响应 - 中心响应
            if self.center_suppress:
                center = self.dw_c(x)
                y = y - torch.tanh(self.beta) * center

            return y

    def __init__(
        self,
        dim: int,
        k_list=(7, 15, 31),
        center_suppress: bool = True,
        use_grn: bool = True,
        use_residual: bool = True
    ):
        super().__init__()

        self.dim = dim
        self.k_list = k_list
        self.num_scales = len(k_list)
        self.use_grn = use_grn
        self.use_residual = use_residual

        # -----------------------------
        # 1) 多尺度外围分支
        # -----------------------------
        self.branches = nn.ModuleList([
            CFPG.Branch(dim, k, center_suppress=center_suppress) for k in k_list
        ])

        # -----------------------------
        # 2) 固定频率滤波器
        # 包括：Sobel X / Sobel Y / Laplace
        # 这些不是可学习参数，而是固定先验
        # -----------------------------
        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        laplace = torch.tensor(
            [[0,  1, 0],
             [1, -4, 1],
             [0,  1, 0]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)
        self.register_buffer("laplace", laplace, persistent=False)

        # -----------------------------
        # 3) 粗门控头
        # 输入是 3 个频率图：
        #   梯度强度、拉普拉斯幅值、局部方差
        # 输出是每个尺度的像素级 logits
        # -----------------------------
        self.coarse_gate = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(16, self.num_scales, kernel_size=1, bias=True)
        )

        # -----------------------------
        # 4) 反馈细化门控头
        # 输入由两部分组成：
        #   a. 原始频率图（3通道）
        #   b. 各尺度真实响应强度图（num_scales 通道）
        # 这样能把“先验频率线索”与“真实分支响应”结合起来
        # -----------------------------
        self.refine_gate = nn.Sequential(
            nn.Conv2d(3 + self.num_scales, 32, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(32, self.num_scales, kernel_size=1, bias=True)
        )

        # -----------------------------
        # 5) 不确定性融合头
        # 用粗门控熵来控制：
        # 当前像素位置应该更相信粗门控，还是更相信反馈细化门控
        # 输出范围在 0~1
        # -----------------------------
        self.confidence_head = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(8, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # -----------------------------
        # 6) 通道重标定
        # 对融合后的结果再做一次通道级增强
        # -----------------------------
        self.channel_reweight = ChannelReweight(dim)

        # -----------------------------
        # 7) 输出投影
        # 用于进一步整合多尺度融合后的特征
        # -----------------------------
        self.out_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        )

        # -----------------------------
        # 8) 可选 GRN
        # -----------------------------
        if self.use_grn:
            self.grn = GRN(dim)

    def _depthwise_filter(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """
        将单个固定 3x3 核应用到所有通道上，使用 depthwise 方式卷积。
        输入:
            x: (B, C, H, W)
            kernel: (1, 1, 3, 3)
        输出:
            (B, C, H, W)
        """
        b, c, h, w = x.shape
        weight = kernel.repeat(c, 1, 1, 1)
        return F.conv2d(x, weight, padding=1, groups=c)

    def _build_freq_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        构建三类频率线索图：
        1. 梯度幅值图：反映边缘和方向变化强度
        2. 拉普拉斯幅值图：反映局部二阶变化
        3. 局部方差图：反映局部纹理复杂度

        最终输出:
            (B, 3, H, W)
        """
        gx = self._depthwise_filter(x, self.sobel_x)
        gy = self._depthwise_filter(x, self.sobel_y)
        lap = self._depthwise_filter(x, self.laplace)

        # 梯度幅值
        grad_mag = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6)

        # 局部方差
        mean = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        mean2 = F.avg_pool2d(x * x, kernel_size=3, stride=1, padding=1)
        var = torch.clamp(mean2 - mean * mean, min=0.0)

        # 沿通道求平均，得到单通道频率描述图
        f1 = grad_mag.mean(dim=1, keepdim=True)
        f2 = lap.abs().mean(dim=1, keepdim=True)
        f3 = var.mean(dim=1, keepdim=True)

        return torch.cat([f1, f2, f3], dim=1)

    def _branch_response_maps(self, branch_outputs):
        """
        根据各尺度分支的真实输出，构建“响应强度图”。

        这里不直接把所有分支特征拼接送入 refine gate，
        而是取每个分支输出的平均绝对值作为响应强度图：
        这样做的好处：
        1. 参数量更小；
        2. 更适合作为“反馈信号”；
        3. 更像一种尺度能量图。

        输入:
            branch_outputs: 长度为 num_scales 的列表，每个元素形状为 (B, C, H, W)

        输出:
            (B, num_scales, H, W)
        """
        energy_maps = []
        for y in branch_outputs:
            # 对每个分支，计算通道平均的绝对响应强度
            e = y.abs().mean(dim=1, keepdim=True)
            energy_maps.append(e)
        return torch.cat(energy_maps, dim=1)

    def _entropy_map(self, prob: torch.Tensor) -> torch.Tensor:
        """
        根据尺度概率分布计算像素级熵图。
        熵越高，说明粗门控越不确定。
        输入:
            prob: (B, K, H, W)，K 为尺度数
        输出:
            (B, 1, H, W)
        """
        entropy = -(prob * torch.log(prob + 1e-8)).sum(dim=1, keepdim=True)

        # 为了使熵值更稳定，归一化到 0~1 左右的范围
        entropy = entropy / torch.log(torch.tensor(float(self.num_scales), device=prob.device))
        return entropy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向流程总览：

        第一步：提取频率线索，生成粗门控
        第二步：计算多尺度外围分支输出
        第三步：根据真实分支响应，生成反馈细化门控
        第四步：根据粗门控熵估计不确定性，融合粗门控与细化门控
        第五步：进行多尺度加权融合
        第六步：通道重标定 + 输出投影 + 残差连接
        """

        # ----------------------------------------
        # 1) 频率线索
        # ----------------------------------------
        freq_maps = self._build_freq_maps(x)  # (B, 3, H, W)

        # ----------------------------------------
        # 2) 粗门控
        # 仅根据频率线索做第一次尺度选择
        # ----------------------------------------
        coarse_logits = self.coarse_gate(freq_maps)                  # (B, K, H, W)
        coarse_alpha = torch.softmax(coarse_logits, dim=1)          # (B, K, H, W)

        # ----------------------------------------
        # 3) 多尺度外围分支
        # ----------------------------------------
        branch_outputs = [branch(x) for branch in self.branches]

        # ----------------------------------------
        # 4) 构建反馈信号
        # 用真实分支响应来反向修正门控
        # ----------------------------------------
        branch_energy = self._branch_response_maps(branch_outputs)   # (B, K, H, W)

        refine_input = torch.cat([freq_maps, branch_energy], dim=1) # (B, 3+K, H, W)
        refine_logits = self.refine_gate(refine_input)               # (B, K, H, W)
        refine_alpha = torch.softmax(refine_logits, dim=1)          # (B, K, H, W)

        # ----------------------------------------
        # 5) 门控不确定性估计
        # 若粗门控熵大，说明不确定性高，则更信任 refine_alpha
        # ----------------------------------------
        entropy_map = self._entropy_map(coarse_alpha)                # (B, 1, H, W)
        refine_ratio = self.confidence_head(entropy_map)             # (B, 1, H, W)

        # 最终门控：
        # 当 refine_ratio 越大，越偏向细化门控
        final_alpha = (1.0 - refine_ratio) * coarse_alpha + refine_ratio * refine_alpha

        # 为了数值更稳定，再归一化一次
        final_alpha = final_alpha / (final_alpha.sum(dim=1, keepdim=True) + 1e-8)

        # ----------------------------------------
        # 6) 多尺度加权融合
        # ----------------------------------------
        y = 0.0
        for i, branch_y in enumerate(branch_outputs):
            y = y + branch_y * final_alpha[:, i:i+1, :, :]

        # ----------------------------------------
        # 7) 通道重标定
        # ----------------------------------------
        y = self.channel_reweight(y)

        # ----------------------------------------
        # 8) 输出投影
        # ----------------------------------------
        y = self.out_proj(y)

        # ----------------------------------------
        # 9) 可选 GRN
        # ----------------------------------------
        if self.use_grn:
            y = self.grn(y)

        # ----------------------------------------
        # 10) 残差连接
        # ----------------------------------------
        if self.use_residual:
            y = y + x

        return y


# -----------------------------
# 直接运行测试
# -----------------------------
if __name__ == "__main__":
    # 自动选择设备
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 构造输入张量
    input_tensor = torch.randn(2, 32, 128, 128).to(device)

    # 实例化模型
    model = CFPG(
        dim=32,
        k_list=(7, 15, 31),
        center_suppress=True,
        use_grn=True,
        use_residual=True
    ).to(device)

    # 打印模型结构
    print("========== CFPG 模块结构 ==========")
    print(model)

    # 前向推理
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("\n========== 维度验证 ==========")
    print("输入张量形状:", input_tensor.shape)
    print("输出张量形状:", output_tensor.shape)

    # 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n========== 参数统计 ==========")
    print(f"总参数量: {total_params / 1e6:.4f} M")
    print(f"可训练参数量: {trainable_params / 1e6:.4f} M")

    # 简单数值检查
    print("\n========== 数值检查 ==========")
    print("输出均值:", output_tensor.mean().item())
    print("输出标准差:", output_tensor.std().item())

    print("\nCFPG 模块运行成功。")