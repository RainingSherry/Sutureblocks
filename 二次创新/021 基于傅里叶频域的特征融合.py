import torch
import torch.nn as nn
import torch.fft as fft
from typing import Tuple
"""
    基于傅里叶变换的自适应频域特征融合：
        写作思路与代码讲解：https://www.bilibili.com/video/BV13hpWzfEo6/
        作用位置：任何两个相同大小的特征融合时，或者任何即插即用模块中。
        主要功能（写作要点）：①增强高频信息（重建边缘、纹理、细节）；
                          ②抑制无效特征（噪声、背景干扰）；
                          ③提升小目标与细长结构的感知能力；
                          ④强化跨模态/跨任务特征融合的鲁棒性。
        代码层面：交叉门控融合 → FFT频域增强 → IFFT空间重建
"""
class FFT2DDecompose(nn.Module):
    """
    将 BCHW 的实数特征做 2D FFT，输出 (real, imag) 两个实数张量，形状均为 BCHW
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, C, H, W] —— 输入的空间域实数特征
        x_fft = fft.fft2(x, dim=(-2, -1))          # 对最后两个维度(H, W)做二维FFT
        real = x_fft.real                           # 提取实部，形状[B, C, H, W]
        imag = x_fft.imag                           # 提取虚部，形状[B, C, H, W]
        return real, imag

class IFFT2DReconstruct(nn.Module):
    """
    将 (real, imag) 两个实数张量重组成复数，再做 2D IFFT，返回实部（BCHW）
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        # real/imag: [B, C, H, W] —— 频域实部与虚部
        x_complex = torch.complex(real, imag)       # 组装成复数张量
        x_ifft = fft.ifft2(x_complex, dim=(-2, -1)) # 二维IFFT回到空间域
        return x_ifft.real                          # 只取实部作为输出

class FrequencyEnhanceBlock(nn.Module):
    """
    频域增强块：空间域→FFT→(实/虚拼接)→逐通道1x1变换→拆分实/虚→IFFT→空间域
    """
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.fft = FFT2DDecompose()                 # 空间→频域
        self.conv = nn.Conv2d(
            in_channels=in_channels * 2,
            out_channels=in_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=in_channels * 2
        )
        self.ifft = IFFT2DReconstruct()             # 频域→空间

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        real, imag = self.fft(x)                    # 得到实部/虚部，各形状[B, C, H, W]
        freq_cat = torch.cat([real, imag], dim=1)   # 拼接为 [B, 2C, H, W]
        freq_out = self.conv(freq_cat)   # 逐通道1x1：形状不变 [B, 2C, H, W]

        # 将输出再一分为二：前C为实部，后C为虚部
        c = freq_out.shape[1] // 2
        real_new = freq_out[:, :c, :, :]
        imag_new = freq_out[:, c:, :, :]

        # IFFT 回到空间域
        x_out = self.ifft(real_new, imag_new)       # [B, C, H, W]
        return x_out


class AttentiveAdaptiveFusion(nn.Module):
    def __init__(self, dim: int, bias: bool = False) -> None:
        super().__init__()

        # 融合后压缩通道数：拼接后通道=2C → 压回 C
        self.fuse_proj = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        # 频域增强
        self.freq_enhance = FrequencyEnhanceBlock(in_channels=dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ch_weight = torch.sigmoid(x)  #  可以对x先加入一个即插即用模块后，再传入 sigmod
        sp_weight = torch.sigmoid(y)  #  可以对y先加入一个即插即用模块后，再传入 sigmod

        x_spatial = x * sp_weight                          # [B, C, H, W]
        y_channel = y * ch_weight                          # [B, C, H, W]

        # 通道维拼接后压缩
        fused = torch.cat([x_spatial, y_channel], dim=1)   # [B, 2C, H, W]
        fused = self.fuse_proj(fused)                      # [B, C, H, W]

        # 频域增强
        out = self.freq_enhance(fused)                     # [B, C, H, W]
        return out

if __name__ == "__main__":
    feat_x = torch.randn(1, 32, 50, 50)
    feat_y = torch.randn(1, 32, 50, 50)
    model = AttentiveAdaptiveFusion(dim=32)
    out = model(feat_x, feat_y)
    print(f"输入张量形状: {feat_x.shape}")
    print(f"输入张量形状: {feat_y.shape}")
    print(f"输出张量形状: {out.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：微创新·代码无误")