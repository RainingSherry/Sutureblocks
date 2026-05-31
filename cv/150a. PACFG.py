import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock_noBN(nn.Module):
    """
    不带 BN 的残差块
    结构：
    Conv -> ReLU -> Conv -> Residual Add
    """
    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


class SEBlock(nn.Module):
    """
    标准 SE 注意力模块
    用于通道自适应重标定
    """
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        mid_channels = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.fc(x)
        return x * scale + x


class MultiConvBlock(nn.Module):
    """
    多尺度卷积块
    设计目的：
    1. 作为频域增强后的空间补偿分支
    2. 提供不同感受野的局部建模能力
    3. 与频域分支形成互补
    """
    def __init__(self, dim, num_heads=4):
        super(MultiConvBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads

        # 先做通道压缩，降低计算量
        self.conv_reduction = nn.Conv2d(dim, dim // 4, kernel_size=1, stride=1, bias=True)
        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)

        # 多尺度深度卷积
        self.local_convs = nn.ModuleList([
            nn.Conv2d(
                dim // 4,
                dim // 4,
                kernel_size=(3 + i * 2),
                padding=(1 + i),
                stride=1,
                groups=dim // 4
            ) for i in range(num_heads)
        ])

        # 多尺度特征融合
        self.conv_fusion = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=True)
        self.se_block = SEBlock(dim)

    def forward(self, x):
        x_reduced = self.leakyrelu(self.conv_reduction(x))

        multi_scale_features = []
        for conv in self.local_convs:
            x_scale = self.leakyrelu(conv(x_reduced))
            # 用输入特征作为门控，增强有效局部响应
            x_scale = x_scale * torch.sigmoid(x_reduced)
            multi_scale_features.append(x_scale)

        x_concat = torch.cat(multi_scale_features, dim=1)
        x_fused = self.conv_fusion(x_concat)
        x_fused = self.se_block(x_fused)

        return x + x_fused


class ChannelAttentionFusion(nn.Module):
    """
    双分支通道注意力融合模块
    用于融合：
    1. 频域增强分支特征
    2. 空域多尺度分支特征
    """
    def __init__(self, nf):
        super(ChannelAttentionFusion, self).__init__()
        hidden = max(nf // 4, 4)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(nf * 2, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, nf * 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, fft_features, multi_features):
        combined_features = torch.cat([fft_features, multi_features], dim=1)
        attention_weights = self.fc(self.global_avg_pool(combined_features))
        fft_weight, multi_weight = torch.split(attention_weights, fft_features.size(1), dim=1)
        fused_features = fft_weight * fft_features + multi_weight * multi_features
        return fused_features


class SpatialContextRefine(nn.Module):
    """
    轻量空间上下文细化模块
    设计思想：
    1. 频域重建后，补充空间局部结构
    2. 使用普通卷积 + 空洞卷积联合建模
    3. 用 SE 做通道重标定
    """
    def __init__(self, nf):
        super(SpatialContextRefine, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=2, dilation=2, bias=True)
        self.conv3 = nn.Conv2d(nf * 2, nf, kernel_size=1, stride=1, padding=0, bias=True)
        self.se = SEBlock(nf)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        residual = x
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(x))
        out = self.conv3(torch.cat([x1, x2], dim=1))
        out = self.se(out)
        return residual + out


class PACFGConsistencyGate(nn.Module):
    """
    PACFG 核心模块：相位对齐跨频引导门控
    新增创新点：
    1. 显式构造“相位一致性图”
    2. 显式构造“幅值一致性图”
    3. 让相位一致性反向指导幅值增强
    4. 让幅值一致性反向指导相位校正

    这样做的好处：
    - 原始 DFGF 更像“先验辅助增强”
    - 这里进一步变成“跨频一致性驱动增强”
    - 更符合 CVPR 风格里常见的 consistency-guided design
    """
    def __init__(self, nf):
        super(PACFGConsistencyGate, self).__init__()

        # 幅值分支：输入 = 当前幅值特征 + 先验幅值特征 + 相位一致性图
        self.amp_gate = nn.Sequential(
            nn.Conv2d(nf * 3, nf, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.amp_bias = nn.Sequential(
            nn.Conv2d(nf * 3, nf, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0, bias=True)
        )

        # 相位分支：输入 = 当前相位特征 + 先验相位特征 + 幅值一致性图
        self.pha_gate = nn.Sequential(
            nn.Conv2d(nf * 3, nf, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.pha_bias = nn.Sequential(
            nn.Conv2d(nf * 3, nf, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Tanh()
        )

    def forward(self, raw_mag, raw_pha, prior_mag, prior_pha, mag_feat, pha_feat):
        """
        输入说明：
        raw_mag   : 当前图像频谱幅值
        raw_pha   : 当前图像频谱相位
        prior_mag : 亮图先验的幅值特征
        prior_pha : 亮图先验的相位特征
        mag_feat  : 当前幅值经卷积编码后的特征
        pha_feat  : 当前相位经卷积编码后的特征
        """

        # -----------------------------
        # 1. 构造相位一致性图
        # -----------------------------
        # cos(Δphase) 越接近 1，说明相位越一致
        # 这里将其映射到 [0,1]，作为可学习引导信号
        phase_consistency = (torch.cos(raw_pha - prior_pha) + 1.0) * 0.5

        # -----------------------------
        # 2. 构造幅值一致性图
        # -----------------------------
        # 幅值差越小，一致性越高
        # 用 exp(-|Δmag|) 构造一个平滑一致性响应
        amplitude_consistency = torch.exp(-torch.abs(raw_mag - prior_mag))

        # -----------------------------
        # 3. 相位一致性反向指导幅值增强
        # -----------------------------
        amp_input = torch.cat([mag_feat, prior_mag, phase_consistency], dim=1)
        amp_gate = self.amp_gate(amp_input)
        amp_bias = self.amp_bias(amp_input)

        # 幅值更新 = 当前幅值 + 门控先验注入 + 学习偏置
        refined_mag = mag_feat + amp_gate * prior_mag + amp_bias

        # -----------------------------
        # 4. 幅值一致性反向指导相位校正
        # -----------------------------
        pha_input = torch.cat([pha_feat, prior_pha, amplitude_consistency], dim=1)
        pha_gate = self.pha_gate(pha_input)
        pha_bias = self.pha_bias(pha_input)

        # 相位更新时用较小系数控制偏移幅度，增强训练稳定性
        refined_pha = pha_feat + pha_gate * prior_pha + 0.5 * pha_bias

        return refined_mag, refined_pha


class PACFGSpectralBlock(nn.Module):
    """
    PACFG 频域增强块
    核心流程：
    1. 空间特征转频域
    2. 幅值/相位解耦建模
    3. 引入 PACFG 一致性跨频门控
    4. 频域重建回空间
    5. 追加空间上下文细化
    """
    def __init__(self, nf):
        super(PACFGSpectralBlock, self).__init__()
        self.nf = nf

        # 频域预处理
        self.freq_preprocess = nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0)

        # 幅值编码
        self.process_amp = nn.Sequential(
            nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0)
        )

        # 相位编码
        self.process_pha = nn.Sequential(
            nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0)
        )

        # PACFG 一致性门控
        self.consistency_gate = PACFGConsistencyGate(nf)

        # 频域重建后的空间细化
        self.spatial_refine = SpatialContextRefine(nf)

    def forward(self, x, prior_mag, prior_pha):
        residual = x
        _, _, H, W = x.shape

        # -----------------------------
        # 1. 转到频域
        # -----------------------------
        x_freq = torch.fft.rfft2(self.freq_preprocess(x), norm='backward')
        raw_mag = torch.abs(x_freq)
        raw_pha = torch.angle(x_freq)

        # -----------------------------
        # 2. 幅值 / 相位编码
        # -----------------------------
        mag_feat = self.process_amp(raw_mag)
        pha_feat = self.process_pha(raw_pha)

        # -----------------------------
        # 3. PACFG 一致性跨频引导
        # -----------------------------
        refined_mag, refined_pha = self.consistency_gate(
            raw_mag=raw_mag,
            raw_pha=raw_pha,
            prior_mag=prior_mag,
            prior_pha=prior_pha,
            mag_feat=mag_feat,
            pha_feat=pha_feat
        )

        # -----------------------------
        # 4. 频域重建
        # -----------------------------
        real = refined_mag * torch.cos(refined_pha)
        imag = refined_mag * torch.sin(refined_pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')

        # -----------------------------
        # 5. 空间上下文细化
        # -----------------------------
        x_out = self.spatial_refine(x_out)

        return residual + x_out


class LowFrequencyProcessing_PACFG(nn.Module):
    """
    PACFG 低频处理主干
    结构设计：
    1. 初始特征提取
    2. 多层 PACFG 频域增强块堆叠
    3. 并行多尺度卷积分支
    4. 通道注意力融合
    5. 残差重建输出
    """
    def __init__(self, nf=32, num_blocks=4, input_channels=3):
        super(LowFrequencyProcessing_PACFG, self).__init__()

        self.initial_conv = nn.Conv2d(input_channels, nf, kernel_size=3, stride=1, padding=1, bias=True)

        # 频域主分支
        self.pacfg_blocks = nn.ModuleList([PACFGSpectralBlock(nf) for _ in range(num_blocks)])

        # 空域补偿分支
        self.multi_blocks = nn.ModuleList([MultiConvBlock(nf) for _ in range(num_blocks)])

        # 融合模块
        self.fusion_block = ChannelAttentionFusion(nf)

        # 重建模块
        self.recon_trunk = ResidualBlock_noBN(nf=nf)
        self.upconv_last = nn.Conv2d(nf, 3, 3, 1, 1, bias=True)

    def forward(self, x, prior_mag, prior_pha):
        x_ori = x
        feat = self.initial_conv(x)

        # 频域分支
        fft_features = feat
        for block in self.pacfg_blocks:
            fft_features = block(fft_features, prior_mag, prior_pha)

        # 空域多尺度分支
        multi_features = feat
        for block in self.multi_blocks:
            multi_features = block(multi_features)

        # 融合
        fused_features = self.fusion_block(fft_features, multi_features)

        # 重建
        fused_features = self.recon_trunk(fused_features)
        out = self.upconv_last(fused_features) + x_ori

        return out


class PACFG_DFGFLow(nn.Module):
    """
    新模块名称：
    PACFG_DFGFLow

    全称：
    Phase-Aligned Cross-Frequency Guidance for Low-Frequency Restoration

    设计定位：
    这是一个可以直接替代原始 DFGFLow 的魔改版本，
    forward 接口仍然保持：
        forward(x, x_light)

    其中：
    x       : 待恢复输入图像
    x_light : 亮图先验 / 引导图像
    """
    def __init__(self, nf=32, num_blocks=4):
        super(PACFG_DFGFLow, self).__init__()
        self.nf = nf

        # 对亮图先验做频域编码
        self.conv_first_mag = nn.Conv2d(3, nf, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_first_pha = nn.Conv2d(3, nf, kernel_size=1, stride=1, padding=0, bias=True)

        # 主处理模块
        self.processblock = LowFrequencyProcessing_PACFG(
            nf=nf,
            num_blocks=num_blocks,
            input_channels=3
        )

    def forward(self, x, x_light):
        """
        输入：
        x       : 原始输入图像，形状 [B, 3, H, W]
        x_light : 引导亮图，形状 [B, 3, H, W]

        输出：
        out     : 恢复结果，形状 [B, 3, H, W]
        """

        # -----------------------------
        # 1. 对引导亮图做频域分解
        # -----------------------------
        x_light_freq = torch.fft.rfft2(x_light, norm='backward')
        x_light_mag = torch.abs(x_light_freq)
        x_light_pha = torch.angle(x_light_freq)

        # -----------------------------
        # 2. 先验编码到特征空间
        # -----------------------------
        prior_mag = self.conv_first_mag(x_light_mag)
        prior_pha = self.conv_first_pha(x_light_pha)

        # -----------------------------
        # 3. 低频增强主干
        # -----------------------------
        out = self.processblock(x, prior_mag, prior_pha)

        return out


if __name__ == "__main__":
    """
    直接运行测试
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 构造两个输入
    # input_tensor_x1 表示待恢复图像
    # input_tensor_x2 表示亮图先验 / 引导图像
    input_tensor_x1 = torch.randn(1, 3, 128, 128).to(device)
    input_tensor_x2 = torch.randn(1, 3, 128, 128).to(device)

    # 实例化模型
    model = PACFG_DFGFLow(nf=32, num_blocks=4).to(device)

    # 前向测试
    output_tensor = model(input_tensor_x1, input_tensor_x2)

    # 打印模型与结果维度
    print(model)
    print("input_tensor_shape_x1 :", input_tensor_x1.shape)
    print("input_tensor_shape_x2 :", input_tensor_x2.shape)
    print("output_tensor_shape   :", output_tensor.shape)

    # 额外打印参数量，方便你写论文或做对比实验
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n模型总参数量: {:.4f} M".format(total_params / 1e6))
    print("可训练参数量: {:.4f} M".format(trainable_params / 1e6))
    print("\nPACFG 模块运行成功！\n")