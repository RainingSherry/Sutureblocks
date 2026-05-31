import torch
import torch.nn as nn
from einops import rearrange

"""
    分块自适应傅里叶频域模块：
        写作思路与代码讲解：https://www.bilibili.com/video/BV1yqaRztEkF/
        作用位置：任何单一输出特征后，或者任何即插即用模块中。
	    主要功能：对图像特征图分块处理，并在频域中通过可学习的权重参数对每个图像频率分量进行自适应调整。
        代码层面：①按指定尺寸分块；②对每个图像块执行二维实数傅里叶变换（RFFT）；
                ③通过可学习参数对频域分量进行缩放调整；④执行逆傅里叶变换（IRFFT）返回空间域；
                ⑤重组图像块恢复原始尺寸。
"""

class PatchwiseSpectralFilter(nn.Module):
    """
        按 patch 在频域做可学习滤波（即插即用）。
        输入/输出形状不变：(B, C, H, W)，要求 H、W 能被 patch_size 整除。
    """
    def __init__(self, channels: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.channels = channels
        # 频域权重：针对每个通道、每个patch频率分量的缩放（实数），与 rfft2 输出尺寸对齐的最后一维 (W//2+1)
        self.freq_weight = nn.Parameter(
            torch.ones(channels, 1, 1, patch_size, patch_size // 2 + 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        ps = self.patch_size
        if (h % ps != 0) or (w % ps != 0):
            raise ValueError(f"H({h}) 与 W({w}) 必须能被 patch_size({ps}) 整除。")

        # 空间按 patch 分块 -> (B, C, H/ps, W/ps, ps, ps)
        x_patch = rearrange(
            x, 'b c (hh ps1) (ww ps2) -> b c hh ww ps1 ps2',
            ps1=ps, ps2=ps
        )

        # 2D 实数 FFT 到频域（复数）
        x_patch_fft = torch.fft.rfft2(x_patch.float())  # complex64

        # 逐通道、逐频率分量可学习缩放（广播到 B、C、hh、ww）
        x_patch_fft = x_patch_fft * self.freq_weight  # 实数权重缩放复数谱

        # 逆 FFT 回到空间域（需指定原始大小）
        x_patch_filtered = torch.fft.irfft2(x_patch_fft, s=(ps, ps))

        # 还原为原图形状 -> (B, C, H, W)
        x_out = rearrange(
            x_patch_filtered, 'b c hh ww ps1 ps2 -> b c (hh ps1) (ww ps2)',
            ps1=ps, ps2=ps
        )
        return x_out.to(dtype=x.dtype, device=x.device)

if __name__ == "__main__":
    x = torch.randn(1, 32, 64, 64)  # H W 需能被 patch_size 整除
    model = PatchwiseSpectralFilter(channels=32, patch_size=8)
    y = model(x)
    print(f"输入张量形状: {x.shape}")
    print(f"输出张量形状: {y.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：微创新·代码无误")