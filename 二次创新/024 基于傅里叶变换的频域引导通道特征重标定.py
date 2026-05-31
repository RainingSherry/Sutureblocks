import torch
import torch.nn as nn

"""
    基于傅里叶变换的频域引导通道特征重标定：
        写作思路与代码讲解：https://www.bilibili.com/video/BV13fQgB5E1g/
        作用位置：任何单一特征处理时，或者任何即插即用模块中。
        主要功能（写作要点）：①通道权重的重标定；②特征冗余抑制与判别增强；③边界与细节保持；
        代码层面：通过频率域权重调整实现对图像中关键频率成分（如边缘、纹理等高频信息或整体结构等低频信息）的自适应关注，提升特征表达能力。
"""


class FrequencyChannelAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        # 1×1卷积调整通道并激活
        self.preprocess_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.GELU()
        )

        # ===== 空间通道注意力（SCA, 定义但未使用，保持与原逻辑一致）=====
        self.sca_conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.sca_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ===== 频域通道注意力（FCA, forward 中实际使用）=====
        self.fca_conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.fca_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # 1) 通道预处理
        features = self.preprocess_conv(x)

        # 2) 通道权重（来自 GAP + 1×1 卷积）
        channel_weights = self.fca_conv(self.fca_pool(features))

        # 3) FFT -> 频域调制 -> IFFT
        freq_features = torch.fft.fft2(features, norm='backward')
        freq_features = channel_weights * freq_features
        recon_features = torch.fft.ifft2(freq_features, dim=(-2, -1), norm='backward')

        # 4) 取幅值 计算张量中每个元素的绝对值
        output = torch.abs(recon_features)
        return output


if __name__ == '__main__':
    input_tensor = torch.randn(1, 32, 50, 50)
    model = FrequencyChannelAttention(channels=32)
    output_tensor = model(input_tensor)
    print(f"输入张量形状: {input_tensor.shape}")
    print(f"输出张量形状: {output_tensor.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")
