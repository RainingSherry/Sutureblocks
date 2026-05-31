# -*- coding: utf-8 -*-
"""
cv缝合救星独家复现 | NS-FPN 原版 LFP 模块即插即用版

说明：
1. 这份代码按作者仓库 NS_FPN.py 里的 LFP 相关实现整理出来。
2. 保留原作者的类名和主要逻辑，方便你后面直接缝进自己的 FPN / U-Net / Neck。
3. 这里复现的是论文中的 LFP，对应仓库里的 wav_Enhance。
4. 依赖 pytorch_wavelets，请先安装：
   pip install pytorch_wavelets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse


class ConvDWT(nn.Module):
    """
    原仓库版 DWT 封装
    输入:  (B, C, H, W)
    输出:  (B, 4C, H/2, W/2)
    其中前 C 通道是低频 LL，后 3C 通道是高频 LH/HL/HH
    """
    def __init__(self, wave='haar', mode='zero'):
        super(ConvDWT, self).__init__()
        self.dwt_forward = DWTForward(J=1, wave=wave, mode=mode)

    def forward(self, x):
        # 原作者这里专门关掉 autocast，避免小波变换在半精度下出奇怪问题
        with torch.cuda.amp.autocast(enabled=False):
            if x.dtype != torch.float32:
                x = x.float()

            Yl, Yh = self.dwt_forward(x)
            b, c, h, w = x.shape

            # Yl: (B, C, H/2, W/2) -> 低频 LL
            # Yh[0]: (B, C, 3, H/2, W/2) -> 高频 LH, HL, HH
            Yh = Yh[0].transpose(1, 2).reshape(
                Yh[0].shape[0],
                -1,
                Yh[0].shape[3],
                Yh[0].shape[4]
            )

            output = torch.cat((Yl, Yh), dim=1)
            output = F.interpolate(
                output,
                size=(h // 2, w // 2),
                mode='bilinear',
                align_corners=False
            )
            return output


class ConvIDWT(nn.Module):
    """
    原仓库版 IDWT 封装
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

        # 按原仓库格式恢复成 (B, C, 3, H, W)
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
    原仓库版空间注意力
    用低频分量生成一张空间权重图，去引导高频分量
    """
    def __init__(self, kernel_size=7, bn_before_sigmoid=False):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'

        padding = 3 if kernel_size == 7 else 1
        self.bn_before_sigmoid = bn_before_sigmoid
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

        if bn_before_sigmoid:
            self.bn = nn.BatchNorm2d(1)
            self.bn.bias.data.fill_(0)
            self.bn.bias.requires_grad = False

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)

        if self.bn_before_sigmoid:
            x = self.bn(x)

        return self.sigmoid(x)


class LearnableGaussianFilterBank(nn.Module):
    def __init__(self, kernel_size, num_filters, num_channels):
        super(LearnableGaussianFilterBank, self).__init__()
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.C = num_channels
        self.padding = kernel_size // 2

        # 每个滤波器一个可学习 sigma
        self.sigmas = nn.ParameterList([
            nn.Parameter(torch.tensor([1.0])) for _ in range(num_filters)
        ])

    def forward(self, x):
        weights = [
            self._gaussian_kernel(self.kernel_size, sigma).repeat(self.C, 1, 1, 1)
            for sigma in self.sigmas
        ]

        filtered_outputs = [
            F.conv2d(
                F.pad(
                    x,
                    (self.padding, self.padding, self.padding, self.padding),
                    mode='replicate'
                ),
                weight.to(x.device),
                groups=self.C
            )
            for weight in weights
        ]

        return torch.cat(filtered_outputs, dim=1)

    def _gaussian_kernel(self, kernel_size, sigma):
        kernel = torch.zeros(1, 1, kernel_size, kernel_size)
        center = kernel_size // 2

        for i in range(kernel_size):
            for j in range(kernel_size):
                kernel[:, :, i, j] = torch.exp(
                    -((i - center) ** 2 + (j - center) ** 2) / (2 * sigma ** 2)
                )

        return kernel / kernel.sum()


class wav_Enhance(nn.Module):
    """
    输入输出通道一致，空间尺寸一致
    """
    def __init__(self, in_channels, wave='haar', mode='symmetric',
                 with_gauss=True, gauss_gate=0.5):
        super(wav_Enhance, self).__init__()
        self.dwt = ConvDWT(wave=wave, mode=mode)
        self.idwt = ConvIDWT(wave=wave, mode=mode)
        self.with_gauss = with_gauss
        self.gauss_gate = gauss_gate
        self.attention = SpatialAttention()

        if self.with_gauss:
            self.gaussian_filter = LearnableGaussianFilterBank(
                kernel_size=3,
                num_filters=1,
                num_channels=3 * in_channels
            )

    def forward(self, x):
        B, C, H, W = x.shape

        # 第一步：DWT 分解
        dwt_out = self.dwt(x)      # (B, 4C, H/2, W/2)

        # 前 C 通道是低频 LL，后 3C 通道是高频
        LL = dwt_out[:, :C, :, :]
        Yh = dwt_out[:, C:, :, :]

        # 第二步：低频引导高频
        att = self.attention(LL)   # (B, 1, H/2, W/2)
        Yh = Yh * att

        # 第三步：高斯门控净化
        if self.with_gauss:
            Yh_blurred = self.gaussian_filter(Yh)
            mask = (Yh.abs() < self.gauss_gate).float()
            Yh = Yh * (1 - mask) + Yh_blurred * mask

        # 第四步：IDWT 重建
        x_rec = self.idwt(LL, Yh)  # (B, C, H, W)

        return x_rec


class LFPPlugAndPlay(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.lfp = wav_Enhance(
            in_channels=channels,
            wave='haar',
            mode='zero',       # 仓库在 NS_FPN 里实例化时用的是 mode='zero'
            with_gauss=True,
            gauss_gate=0.5
        )

    def forward(self, x):
        return self.lfp(x)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    # CV 缝合救星提示：视觉特征一般是 (batch, channels, height, width)
    dummy_input = torch.randn(2, 64, 256, 256)  # batch=2, 64通道, 256x256

    # 初始化 LFP 模块
    lfp = wav_Enhance(
        in_channels=64,
        wave='haar',
        mode='zero',
        with_gauss=True,
        gauss_gate=0.5
    )

    print("=== CV 缝合救星 | LFP 模块结构 ===")
    print(lfp)

    output = lfp(dummy_input)
    print("\n=== 输入形状 ===", dummy_input.shape)
    print("=== 输出形状 ===", output.shape)