# -*- coding: utf-8 -*-
"""
CV 缝合救星独家魔改 | PAFP 模块（Prior-Adaptive Frequency Purification）
中文名：先验自适应频率净化模块

设计目标：
1. 保留原始 LFP 的主线：DWT 分解 -> 低频引导高频 -> 高频净化 -> IDWT 重建
2. 在原始 LFP 基础上做更像 CVPR 风格的升级：
   - 从“单一空间引导”升级为“空间 + 通道”的双先验引导
   - 从“固定阈值门控”升级为“内容自适应动态门控”
   - 从“单一高斯滤波”升级为“多尺度高斯专家融合”
   - 在重建后增加轻量残差细化，减少过平滑问题

依赖：
    pip install torch pytorch_wavelets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse


class ConvDWT(nn.Module):
    """
    小波分解模块
    输入:
        x: (B, C, H, W)
    输出:
        output: (B, 4C, H/2, W/2)
    说明:
        前 C 通道是低频 LL
        后 3C 通道是高频 LH / HL / HH
    """
    def __init__(self, wave='haar', mode='zero'):
        super(ConvDWT, self).__init__()
        self.dwt_forward = DWTForward(J=1, wave=wave, mode=mode)

    def forward(self, x):
        # 为了避免半精度在小波变换时带来数值不稳定，这里显式关闭 autocast
        with torch.cuda.amp.autocast(enabled=False):
            if x.dtype != torch.float32:
                x = x.float()

            Yl, Yh = self.dwt_forward(x)
            b, c, h, w = x.shape

            # Yh[0] 的原始形状是 (B, C, 3, H/2, W/2)
            # 这里把 3 个高频子带并到通道维上，变成 (B, 3C, H/2, W/2)
            Yh = Yh[0].transpose(1, 2).reshape(
                Yh[0].shape[0],
                -1,
                Yh[0].shape[3],
                Yh[0].shape[4]
            )

            output = torch.cat((Yl, Yh), dim=1)

            # 这里保持和你原始代码风格一致
            output = F.interpolate(
                output,
                size=(h // 2, w // 2),
                mode='bilinear',
                align_corners=False
            )
            return output


class ConvIDWT(nn.Module):
    """
    小波重建模块
    输入:
        low_freqs:  (B, C, H/2, W/2)
        high_freqs: (B, 3C, H/2, W/2)
    输出:
        reconstruction: (B, C, H, W)
    """
    def __init__(self, wave='haar', mode='zero'):
        super(ConvIDWT, self).__init__()
        self.dwt_inverse = DWTInverse(wave=wave, mode=mode)

    def forward(self, low_freqs, high_freqs):
        B, C, H, W = low_freqs.shape

        # 恢复成 pytorch_wavelets 需要的格式: (B, C, 3, H, W)
        high_freqs = high_freqs.reshape(B, C, 3, H, W)

        with torch.cuda.amp.autocast(enabled=False):
            reconstruction = self.dwt_inverse((low_freqs, [high_freqs.float()]))

            reconstruction = F.interpolate(
                reconstruction,
                size=(2 * H, 2 * W),
                mode='bilinear',
                align_corners=False
            )
            return reconstruction


class SpatialAttention(nn.Module):
    """
    空间注意力
    作用：
        从低频特征中提取空间先验，告诉模型“哪里更像目标区域”
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'

        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ChannelAttention(nn.Module):
    """
    通道注意力
    作用：
        从低频特征中提取通道先验，告诉模型“哪些高频通道更值得保留”
    """
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        hidden = max(in_channels // reduction, 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        att = self.sigmoid(avg_out + max_out)
        return att


class DynamicGatePredictor(nn.Module):
    """
    动态门控阈值预测器
    作用：
        原版 LFP 使用固定 gauss_gate，例如 0.5
        这里改成根据低频特征自适应预测当前样本的净化门限
    输出：
        gate: (B, 1, 1, 1)，范围在 [min_gate, max_gate]
    """
    def __init__(self, in_channels, hidden_ratio=4, min_gate=0.1, max_gate=0.9):
        super(DynamicGatePredictor, self).__init__()
        hidden = max(in_channels // hidden_ratio, 4)
        self.min_gate = min_gate
        self.max_gate = max_gate

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.predictor = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        gate = self.predictor(self.pool(x))
        gate = self.min_gate + (self.max_gate - self.min_gate) * gate
        return gate


class GaussianBlur(nn.Module):
    """
    单个高斯滤波器
    说明：
        这里保留“可学习 sigma”的设计
        每次 forward 时根据 sigma 动态生成核
    """
    def __init__(self, kernel_size, num_channels):
        super(GaussianBlur, self).__init__()
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.padding = kernel_size // 2
        self.sigma = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))

    def _build_kernel(self, device, dtype):
        kernel = torch.zeros(1, 1, self.kernel_size, self.kernel_size, device=device, dtype=dtype)
        center = self.kernel_size // 2
        sigma = torch.clamp(self.sigma, min=0.1)

        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                kernel[:, :, i, j] = torch.exp(
                    -((i - center) ** 2 + (j - center) ** 2) / (2 * sigma ** 2)
                )

        kernel = kernel / (kernel.sum() + 1e-6)
        kernel = kernel.repeat(self.num_channels, 1, 1, 1)
        return kernel

    def forward(self, x):
        kernel = self._build_kernel(x.device, x.dtype)
        out = F.conv2d(
            F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='replicate'),
            kernel,
            groups=self.num_channels
        )
        return out


class MultiScaleGaussianExpert(nn.Module):
    """
    多尺度高斯专家模块
    作用：
        不再只用单一尺度高斯滤波
        而是使用 3x3 和 5x5 两个高斯专家，再根据低频先验预测融合权重
    """
    def __init__(self, num_channels, prior_channels):
        super(MultiScaleGaussianExpert, self).__init__()
        self.blur3 = GaussianBlur(kernel_size=3, num_channels=num_channels)
        self.blur5 = GaussianBlur(kernel_size=5, num_channels=num_channels)

        hidden = max(prior_channels // 4, 4)
        self.weight_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(prior_channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, kernel_size=1, bias=True)
        )

    def forward(self, x, prior):
        # 两个高斯专家分别处理
        out3 = self.blur3(x)
        out5 = self.blur5(x)

        # 根据低频先验预测两个专家的融合权重
        weights = self.weight_predictor(prior)      # (B, 2, 1, 1)
        weights = F.softmax(weights, dim=1)

        w3 = weights[:, 0:1, :, :]
        w5 = weights[:, 1:2, :, :]

        out = out3 * w3 + out5 * w5
        return out


class ResidualRefine(nn.Module):
    """
    轻量残差细化模块
    作用：
        在 IDWT 重建后做一次轻量修正，缓解频域重建后的过平滑问题
    """
    def __init__(self, in_channels):
        super(ResidualRefine, self).__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.GELU()
        )

    def forward(self, x):
        return x + self.refine(x)


class PAFP(nn.Module):
    """
    PAFP: Prior-Adaptive Frequency Purification
    先验自适应频率净化模块

    相比原始 LFP 的核心升级：
    1. 低频不再只生成空间注意力，还生成通道注意力
    2. 净化门限不再固定，而是根据低频内容动态预测
    3. 高频平滑不再只有单一高斯，而是多尺度高斯专家融合
    4. 重建后增加轻量残差细化

    输入输出：
        输入:  (B, C, H, W)
        输出:  (B, C, H, W)
    """
    def __init__(self, in_channels, wave='haar', mode='zero'):
        super(PAFP, self).__init__()
        self.in_channels = in_channels

        # 小波分解与重建
        self.dwt = ConvDWT(wave=wave, mode=mode)
        self.idwt = ConvIDWT(wave=wave, mode=mode)

        # 双先验引导：空间先验 + 通道先验
        self.spatial_attention = SpatialAttention(kernel_size=7)
        self.channel_attention = ChannelAttention(in_channels=in_channels, reduction=16)

        # 动态门控阈值预测器
        self.dynamic_gate = DynamicGatePredictor(
            in_channels=in_channels,
            hidden_ratio=4,
            min_gate=0.1,
            max_gate=0.9
        )

        # 多尺度高斯专家，用来替代原版单一高斯滤波
        self.multi_gaussian = MultiScaleGaussianExpert(
            num_channels=3 * in_channels,
            prior_channels=in_channels
        )

        # 高频融合后的轻量卷积校准
        self.high_freq_fuse = nn.Sequential(
            nn.Conv2d(3 * in_channels, 3 * in_channels, kernel_size=1, bias=False),
            nn.GELU()
        )

        # 重建后的残差细化
        self.refine = ResidualRefine(in_channels=in_channels)

    def forward(self, x):
        B, C, H, W = x.shape

        # =====================================================
        # 第一步：做 DWT 分解
        # 得到低频 LL 和高频 Yh
        # =====================================================
        dwt_out = self.dwt(x)
        LL = dwt_out[:, :C, :, :]
        Yh = dwt_out[:, C:, :, :]

        # =====================================================
        # 第二步：从低频中提取双先验
        # 1）空间先验：强调可能的目标区域
        # 2）通道先验：强调重要的频率通道
        # =====================================================
        spatial_prior = self.spatial_attention(LL)              # (B,1,H/2,W/2)
        channel_prior = self.channel_attention(LL)             # (B,C,H/2,W/2) -> 实际是 (B,C,1,1)

        # 高频有 3C 通道，分别对应 LH / HL / HH
        # 所以把低频通道先验复制 3 份，对齐高频通道数
        channel_prior_hf = torch.cat([channel_prior, channel_prior, channel_prior], dim=1)

        # 双先验联合调制高频
        Yh_guided = Yh * spatial_prior * channel_prior_hf

        # =====================================================
        # 第三步：根据低频内容动态预测门控阈值
        # 原版是固定 gauss_gate，这里改成自适应门限
        # =====================================================
        dynamic_gate = self.dynamic_gate(LL)                   # (B,1,1,1)

        # =====================================================
        # 第四步：多尺度高斯专家净化高频
        # 让不同尺度的高斯滤波共同参与，增强适应性
        # =====================================================
        Yh_blurred = self.multi_gaussian(Yh_guided, LL)

        # 构造动态门控掩码
        # 响应弱的位置更多使用平滑结果
        # 响应强的位置更多保留原始高频
        mask = (Yh_guided.abs() < dynamic_gate).float()

        Yh_purified = Yh_guided * (1 - mask) + Yh_blurred * mask
        Yh_purified = self.high_freq_fuse(Yh_purified)

        # =====================================================
        # 第五步：IDWT 重建回空间域
        # =====================================================
        x_rec = self.idwt(LL, Yh_purified)

        # =====================================================
        # 第六步：重建后再做一次轻量残差细化
        # 让输出更稳一点，减少过平滑
        # =====================================================
        out = self.refine(x_rec)

        return out


if __name__ == "__main__":
    # CV 缝合救星提示：视觉特征一般是 (batch, channels, height, width)
    dummy_input = torch.randn(2, 64, 256, 256)  # batch=2, 64通道, 256x256

    # 初始化 PAFP 模块
    pafp = PAFP(
        in_channels=64,
        wave='haar',
        mode='zero'
    )

    print("=== CV 缝合救星 | PAFP 模块结构 ===")
    print(pafp)

    output = pafp(dummy_input)
    print("\n=== 输入形状 ===", dummy_input.shape)
    print("=== 输出形状 ===", output.shape)