import torch
import math
import torch.nn as nn

"""
    基于高斯滤波的空间注意力模块：
        写作思路与代码讲解：https://www.bilibili.com/video/BV1MrhuzNEgg/
        作用位置：任何单一输出特征后，或者任何即插即用模块中。
	    主要功能：将高斯滤波与注意力机制相结合，通过高斯核生成注意力权重，对特征进行自适应增强。
        代码层面：首先，通过可配置高斯卷积核生成高斯滤波结果，接着经过归一化和 GELU 激活函数生成注意力权重，最终通过逐元素相乘实现对输入特征的自适应调整。
"""

def build_normalization(norm_config, num_features):
    """构建归一化层"""
    norm_type = norm_config.get('type', 'BN')
    requires_grad = norm_config.get('requires_grad', True)

    if norm_type == 'BN':
        norm_layer = nn.BatchNorm2d(num_features)
    else:
        raise NotImplementedError(f"不支持的归一化类型: {norm_type}")

    for param in norm_layer.parameters():
        param.requires_grad = requires_grad

    return norm_type, norm_layer

class GaussianAttention(nn.Module):
    """高斯注意力模块: 结合高斯滤波与注意力机制的特征增强模块"""
    """
    Args:
        channels: 输入特征图的通道数
        kernel_size: 高斯核大小（奇数）
        sigma: 高斯分布的标准差
    # 这里有两个超参数，一个是卷积核大小，一个是初始化高斯卷积的超参数
    # 大家根据实验结果灵活调整
    """
    def __init__(self, channels, kernel_size=5, sigma=1.0):
        super().__init__()
        # 配置归一化层参数
        norm_cfg = dict(type='BN', requires_grad=True)

        # 生成高斯核并初始化卷积层
        gaussian_kernel = self.create_gaussian_kernel(kernel_size, sigma)
        gaussian_kernel = nn.Parameter(gaussian_kernel, requires_grad=False).clone()

        # 初始化分组卷积实现高斯滤波
        self.gaussian_filter = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=False
        )
        self.gaussian_filter.weight.data = gaussian_kernel.repeat(channels, 1, 1, 1)

        # 初始化归一化层和激活函数
        self.norm = build_normalization(norm_cfg, channels)[1]
        self.activation = nn.GELU()

    def create_gaussian_kernel(self, kernel_size, sigma):
        """生成高斯核矩阵"""
        return torch.FloatTensor([
            [(1 / (2 * math.pi * sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
             for x in range(-kernel_size // 2 + 1, kernel_size // 2 + 1)]
            for y in range(-kernel_size // 2 + 1, kernel_size // 2 + 1)
        ]).unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        """前向传播: 生成注意力权重并与输入特征相乘"""
        filtered = self.gaussian_filter(x)
        attention = self.activation(self.norm(filtered))
        return x * attention

if __name__ == "__main__":
    x = torch.randn(1, 32, 50, 50)
    model = GaussianAttention(channels=32)
    output = model(x)
    print(f"输入张量形状: {x.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：微创新·代码无误")