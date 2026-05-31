import torch
import torch.nn as nn

"""
    二维快速傅里叶变换模块：
        写作思路与代码讲解：https://www.bilibili.com/video/BV1WdeuziECo/
        作用位置：任何单一输出特征后，或者任何即插即用模块中。
        主要功能：增强全局频率信息的捕捉能力，卷积操作实现空间域与频域信息融合，提升特征丰富性。
        代码层面：①对输入特征图进行归一化，确保数值稳定性。
                ②二维快速傅里叶变换（FFT）将特征从空间域转换至频域，得到复数形式的频域张量，使用 1×1 卷积进行频域特征变换，并通过 GELU 激活函数引入非线性；
                ③通过逆快速傅里叶变换（IFFT）将特征从频域转换回空间域，输出增强后的特征图。
"""

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class FourierUnit(nn.Module):
    def __init__(self, dim, groups=1, fft_norm='ortho'):
        super().__init__()
        self.groups = groups
        self.fft_norm = fft_norm

        # 处理实部和虚部拼接后的2x通道
        self.freq_conv = nn.Conv2d(
            in_channels=dim * 2,
            out_channels=dim * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=self.groups,
            bias=False
        )
        self.activation = nn.GELU()
        self.norm = LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        # 空间域特征规范化
        x = self.norm(x)
        batch, channels, height, width = x.shape

        # 1. 空间域到频域的转换
        freq_features = torch.fft.rfft2(x, norm=self.fft_norm)  # [B, C, H, W//2 + 1], complex

        # 2. 拆分复数分量并拼接（实部+虚部）
        freq_real = freq_features.real
        freq_imag = freq_features.imag
        freq_combined = torch.cat([freq_real, freq_imag], dim=1)  # [B, 2C, H, W//2+1]

        # 3. 频域特征转换与激活
        freq_transformed = self.freq_conv(freq_combined)
        freq_transformed = self.activation(freq_transformed)

        # 4. 重组复数分量
        out_real, out_imag = torch.chunk(freq_transformed, 2, dim=1)
        freq_complex = torch.complex(out_real, out_imag)  # [B, C, H, W//2+1], complex

        # 5. 频域到空间域的转换
        output = torch.fft.irfft2(freq_complex, s=(height, width), norm=self.fft_norm)  # [B, C, H, W]
        return output

if __name__ == "__main__":
    x = torch.randn(1, 32, 50, 50)
    model = FourierUnit(dim=32)
    output = model(x)
    print(f"输入张量形状: {x.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：微创新·代码无误")