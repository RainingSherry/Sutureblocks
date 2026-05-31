import math
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F


class HRGA4D(nn.Module):
    """
    HRGA4D: Hierarchical Relational-Gated 4D Attention

    模块设计目标：
    1. 保留 Attention4D 的全局关系建模能力
    2. 增强不同注意力头之间的自适应分工
    3. 用双尺度局部分支补强局部纹理与中尺度结构信息
    4. 在输出阶段增加内容感知的空间门控，提升显著区域响应
    """

    def __init__(
        self,
        dim=384,
        key_dim=32,
        num_heads=8,
        attn_ratio=4,
        resolution=7,
        act_layer=nn.ReLU,
        stride=None,
        gate_reduction=4
    ):
        super().__init__()

        # =========================
        # 基础超参数
        # =========================
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.scale = key_dim ** -0.5
        self.attn_ratio = attn_ratio

        # 所有 head 的 key 总维度
        self.nh_kd = key_dim * num_heads

        # 单个 head 的 value 维度
        self.d = int(attn_ratio * key_dim)

        # 所有 head 的 value 总维度
        self.dh = self.d * num_heads

        # =========================
        # 下采样设置
        # 如果 stride 不为空，则先做一次空间降采样
        # =========================
        if stride is not None:
            self.resolution = math.ceil(resolution / stride)

            # 深度卷积做轻量下采样
            self.stride_conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, stride=stride, padding=1, groups=dim, bias=False),
                nn.BatchNorm2d(dim)
            )

            # 输出端恢复原始分辨率
            self.upsample = nn.Upsample(scale_factor=stride, mode="bilinear", align_corners=False)
        else:
            self.resolution = resolution
            self.stride_conv = None
            self.upsample = None

        # 当前注意力操作内部使用的 token 数
        self.N = self.resolution ** 2

        # =========================
        # Q / K / V 投影
        # =========================
        self.q = nn.Sequential(
            nn.Conv2d(dim, self.num_heads * self.key_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.num_heads * self.key_dim)
        )

        self.k = nn.Sequential(
            nn.Conv2d(dim, self.num_heads * self.key_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.num_heads * self.key_dim)
        )

        self.v = nn.Sequential(
            nn.Conv2d(dim, self.num_heads * self.d, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.num_heads * self.d)
        )

        # =========================
        # 双尺度局部分支
        # 3x3 分支偏向局部细节
        # 5x5 分支偏向中尺度结构
        # =========================
        self.v_local_3 = nn.Sequential(
            nn.Conv2d(self.num_heads * self.d, self.num_heads * self.d,
                      kernel_size=3, stride=1, padding=1,
                      groups=self.num_heads * self.d, bias=False),
            nn.BatchNorm2d(self.num_heads * self.d)
        )

        self.v_local_5 = nn.Sequential(
            nn.Conv2d(self.num_heads * self.d, self.num_heads * self.d,
                      kernel_size=5, stride=1, padding=2,
                      groups=self.num_heads * self.d, bias=False),
            nn.BatchNorm2d(self.num_heads * self.d)
        )

        # =========================
        # 双尺度局部融合门控
        # 输入是全局池化后的特征
        # 输出两个权重，对应 3x3 / 5x5 分支
        # =========================
        hidden_dim = max(dim // gate_reduction, 16)
        self.local_scale_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 2, kernel_size=1, bias=True)
        )

        # =========================
        # Talking-Head 机制
        # 对注意力图在 head 维度做轻量交互
        # =========================
        self.talking_head1 = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1, bias=True)
        self.talking_head2 = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1, bias=True)

        # =========================
        # 头级动态门控
        # 根据输入内容自适应调整每个注意力头的重要性
        # 输出形状: (B, num_heads, 1, 1)
        # =========================
        self.head_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_heads, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # =========================
        # 内容空间门
        # 对全局注意力输出做一次空间位置重标定
        # 作用：突出内容敏感区域，抑制背景区域
        # =========================
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(self.dh, self.dh, kernel_size=3, padding=1, groups=self.dh, bias=False),
            nn.BatchNorm2d(self.dh),
            nn.Conv2d(self.dh, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # =========================
        # 输出投影
        # =========================
        self.proj = nn.Sequential(
            act_layer(inplace=True) if act_layer in [nn.ReLU, nn.ReLU6, nn.LeakyReLU] else act_layer(),
            nn.Conv2d(self.dh, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim)
        )

        # =========================
        # 相对位置偏置
        # 这里沿用二维相对偏置的思路
        # 不同空间偏移共享一个 bias 编号
        # =========================
        points = list(itertools.product(range(self.resolution), range(self.resolution)))
        attention_offsets = {}
        idxs = []

        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])

        self.attention_biases = nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets))
        )

        self.register_buffer(
            "attention_bias_idxs",
            torch.LongTensor(idxs).view(self.N, self.N)
        )

    @torch.no_grad()
    def train(self, mode=True):
        """
        训练与推理阶段切换时，缓存 bias 展开结果。
        这样在 eval 阶段可以少做一次重复索引。
        """
        super().train(mode)
        if mode and hasattr(self, "ab"):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        """
        输入:
            x: (B, C, H, W)

        输出:
            out: (B, C, H, W)
        """
        B, C, H, W = x.shape

        # 保留一份输入用于门控分支
        gate_source = x

        # =========================
        # 可选下采样
        # =========================
        if self.stride_conv is not None:
            x = self.stride_conv(x)

        # 当前内部空间大小
        _, _, h_attn, w_attn = x.shape

        # 安全检查：输入分辨率需要与初始化时的设定匹配
        if h_attn != self.resolution or w_attn != self.resolution:
            raise ValueError(
                f"当前输入经过 stride 后的空间尺寸为 {(h_attn, w_attn)}，"
                f"但模块初始化时设定的 resolution 为 {self.resolution}。"
                f"请保证输入大小与 resolution 参数一致。"
            )

        # =========================
        # 生成 Q / K / V
        # =========================
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # q: (B, heads, N, key_dim)
        q = q.flatten(2).reshape(B, self.num_heads, self.key_dim, self.N).permute(0, 1, 3, 2)

        # k: (B, heads, key_dim, N)
        k = k.flatten(2).reshape(B, self.num_heads, self.key_dim, self.N)

        # v: (B, heads, N, d)
        v_attn = v.flatten(2).reshape(B, self.num_heads, self.d, self.N).permute(0, 1, 3, 2)

        # =========================
        # 双尺度局部增强分支
        # =========================
        v_local_3 = self.v_local_3(v)
        v_local_5 = self.v_local_5(v)

        # 根据输入内容自适应决定 3x3 和 5x5 的融合比例
        # local_gate_logits: (B, 2, 1, 1)
        local_gate_logits = self.local_scale_gate(gate_source)

        # 在分支维度上做 softmax，保证两个权重和为 1
        local_gate = F.softmax(local_gate_logits.view(B, 2), dim=1).view(B, 2, 1, 1, 1)

        # 双尺度局部特征加权融合
        v_local = local_gate[:, 0] * v_local_3 + local_gate[:, 1] * v_local_5

        # =========================
        # 全局注意力图
        # attn: (B, heads, N, N)
        # =========================
        rel_bias = self.attention_biases[:, self.attention_bias_idxs] if self.training else self.ab
        attn = (q @ k) * self.scale + rel_bias.unsqueeze(0)

        # Talking-Head 前变换
        attn = self.talking_head1(attn)

        # =========================
        # 头级动态门控
        # 让不同 head 对当前输入做自适应响应
        # =========================
        head_gate = self.head_gate(gate_source)  # (B, heads, 1, 1)
        attn = attn * head_gate

        # 归一化得到注意力权重
        attn = attn.softmax(dim=-1)

        # Talking-Head 后变换
        attn = self.talking_head2(attn)

        # =========================
        # 全局注意力输出
        # x_global: (B, heads, N, d)
        # =========================
        x_global = attn @ v_attn

        # 恢复为空间特征图: (B, dh, H_attn, W_attn)
        x_global = x_global.transpose(2, 3).reshape(B, self.dh, self.resolution, self.resolution)

        # =========================
        # 内容空间门
        # 进一步强调空间上更重要的位置
        # =========================
        spatial_weight = self.spatial_gate(x_global)
        x_global = x_global * spatial_weight

        # =========================
        # 融合全局分支与局部分支
        # =========================
        out = x_global + v_local

        # 如果前面做过下采样，这里恢复回原分辨率
        if self.upsample is not None:
            out = self.upsample(out)

        # 输出投影，恢复到输入通道数
        out = self.proj(out)

        return out


# =========================
# 简单的网络封装示例
# 便于直接替换到主干网络或 neck 中
# =========================
class HRGA4DBlock(nn.Module):
    """
    带残差连接的标准块封装
    """
    def __init__(
        self,
        dim,
        key_dim=16,
        num_heads=8,
        attn_ratio=2,
        resolution=32,
        stride=None,
        mlp_ratio=2.0
    ):
        super().__init__()

        self.attn = HRGA4D(
            dim=dim,
            key_dim=key_dim,
            num_heads=num_heads,
            attn_ratio=attn_ratio,
            resolution=resolution,
            stride=stride
        )

        hidden_dim = int(dim * mlp_ratio)

        # 这里额外加一个轻量前馈网络
        # 让模块更像视觉骨干中的标准 Transformer-style block
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        # 第一层残差：注意力增强
        x = x + self.attn(x)

        # 第二层残差：通道混合
        x = x + self.ffn(x)

        return x


if __name__ == "__main__":
    # 自动选择设备
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 构造随机输入
    # 这里输入分辨率为 32x32，因此 resolution 也设置成 32
    input_tensor = torch.randn(2, 64, 32, 32).to(device)

    # 创建模块
    model = HRGA4DBlock(
        dim=64,
        key_dim=16,
        num_heads=4,
        attn_ratio=2,
        resolution=32,
        stride=None,
        mlp_ratio=2.0
    ).to(device)

    # 切换到训练模式做一次前向
    model.train()
    output_tensor = model(input_tensor)

    print("模型结构：")
    print(model)
    print("\n输入张量形状：", input_tensor.shape)
    print("输出张量形状：", output_tensor.shape)

    # 再切换到推理模式验证一次
    model.eval()
    with torch.no_grad():
        output_eval = model(input_tensor)

    print("推理阶段输出张量形状：", output_eval.shape)
    print("\nHRGA4D 模块运行正常。")
    print("\n哔哩哔哩/微信公众号: CV缝合救星, 独家整理! \n")