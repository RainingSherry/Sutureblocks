import torch
import torch.nn as nn

""" 
   基于中心偏移的风车式卷积模块：
        写作思路与代码讲解：https://www.bilibili.com/video/BV171wMzrErs/
        作用位置：任何单一特征处理时/任何普通卷积，或者任何即插即用模块中。
        主要功能（写作要点）：①处于图像边界上的目标上指标表现一般。②特征容易受背景纹理、噪声干扰。③单一采样中心导致边界连续性不稳定。（将在本视频的写作部分展开阐述）
        代码层面：通过非对称 padding 和水平/垂直卷积分四个方向提取偏移特征，让卷积的“有效中心”分别向上下左右发生偏移，再融合生成丰富的局部方向特征。
"""

def calculate_padding(kernel_size, padding=None, dilation=1):
    """根据卷积核大小、膨胀率计算填充大小，以保持输出形状与输入相同。"""
    if dilation > 1:
        kernel = dilation * (kernel_size - 1) + 1 if isinstance(kernel_size, int) else [
            dilation * (x - 1) + 1 for x in kernel_size]
    else:
        kernel = kernel_size
    if padding is None:
        padding = kernel // 2 if isinstance(kernel, int) else [x // 2 for x in kernel]
    return padding

class StandardConv(nn.Module):
    """标准卷积层，包含卷积、批归一化和激活函数。"""
    default_activation = nn.SiLU()  # 默认激活函数为SiLU

    def __init__(self, in_channels, out_channels, kernel=1, stride=1, padding=None, groups=1, dilation=1,
                 activation=True):
        """初始化卷积层，参数包括输入通道数、输出通道数、卷积核大小、步幅、填充、分组、膨胀率和激活函数。"""
        super().__init__()
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel, stride,
                                     calculate_padding(kernel, padding, dilation),
                                     groups=groups, dilation=dilation, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation_func = self.default_activation if activation is True else \
            activation if isinstance(activation, nn.Module) else nn.Identity()

    def forward(self, x):
        """前向传播，对输入张量应用卷积、批归一化和激活函数。"""
        return self.activation_func(self.batch_norm(self.convolution(x)))

    def optimized_forward(self, x):
        """融合前向传播，只应用卷积和激活函数，用于模型推理优化。"""
        return self.activation_func(self.convolution(x))


class WindmillConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()

        # 非对称填充配置：分别指定左、右、上、下四条边的填充量
        asymmetric_padding = [(kernel_size, 0, 1, 0), (0, kernel_size, 0, 1),
                              (0, 1, kernel_size, 0), (1, 0, 0, kernel_size)]
        self.padding_layers = [nn.ZeroPad2d(padding) for padding in asymmetric_padding]

        self.horizontal_conv = StandardConv(in_channels, out_channels // 4, (1, kernel_size), stride=stride, padding=0)
        self.vertical_conv = StandardConv(in_channels, out_channels // 4, (kernel_size, 1), stride=stride, padding=0)

        self.combine_conv = StandardConv(out_channels, out_channels, 2, stride=1, padding=0)

    def forward(self, x):
        # 卷积核的“有效中心”在特征图上，分别向左 / 右 / 上 / 下发生了偏移。
        # 通过非对称 padding，使得“同一个卷积核在输入坐标系中的对齐中心发生了相对位移”
        # 让卷积的“有效中心”分别向上下左右发生偏移，只不过这种偏移是通过非对称 padding 在输入坐标系中实现的，而不是显式移动卷积核本身。

        # 水平方向两种填充方式的卷积
        h_conv1 = self.horizontal_conv(self.padding_layers[0](x))
        h_conv2 = self.horizontal_conv(self.padding_layers[1](x))

        # 垂直方向两种填充方式的卷积
        v_conv1 = self.vertical_conv(self.padding_layers[2](x))
        v_conv2 = self.vertical_conv(self.padding_layers[3](x))

        # 合并四个方向的结果并进行通道融合
        combined = torch.cat([h_conv1, h_conv2, v_conv1, v_conv2], dim=1)
        return self.combine_conv(combined)

if __name__ == "__main__":
    model = WindmillConv(in_channels=64, out_channels=64)
    input_tensor = torch.randn(1, 64, 50, 50)
    output_tensor = model(input_tensor)
    print(f'Input shape: {input_tensor.size()}')
    print(f'Output shape: {output_tensor.size()}')
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：微创新·代码无误")