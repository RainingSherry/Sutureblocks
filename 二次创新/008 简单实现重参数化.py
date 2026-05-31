import torch
import torch.nn.functional as F
from torch import nn

"""
    简单实现重参数化技巧：
        写作思路与代码讲解：https://www.bilibili.com/video/BV1KcEMz8Ew2/
        作用位置：任何即插即用模块上的卷积部分/注意力部分。
        主要功能：训练时多分支（多个卷积层）增强表征，推理时单分支（一个卷积层）轻量化，为实际场景（如移动端、边缘计算）提供高效解决方案。
        论文关键词：重参数化（Reparameterization）
        代码使用方式与写作思路请务必看视频~
"""
class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        # 定义卷积层，设置输入通道、输出通道、核大小、步幅、填充和偏置
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=bias)
        # 定义批归一化层，适用于输出通道数
        self.norm = nn.BatchNorm2d(ch_out)
        # 定义激活函数，这里使用ReLU
        self.act = nn.ReLU()

    def forward(self, x):
        # 前向传播：卷积 -> 归一化 -> 激活
        return self.act(self.norm(self.conv(x)))

class RepVggBlock(nn.Module):
    # https://arxiv.org/pdf/2101.03697.pdf
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        # 定义3x3卷积层，步幅为1，填充为1
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        # 定义1x1卷积层，步幅为1，无填充
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        # 获取激活函数，这里使用ReLU
        self.act = nn.ReLU()

    def forward(self, x):
        # hasattr : 检查对象是否包含某个属性
        if hasattr(self, 'conv'):
            # 部署模型/推理模式：仅一个卷积（轻量化）
            y = self.conv(x)  # 这里二次改进换卷积 稀疏注意力
        else:
            # 前向传播：两个卷积的结果相加
            y = self.conv1(x) + self.conv2(x) # 这里二次改进换卷积  注意力
        return self.act(y)

    def convert_to_deploy(self):
        # 转换为部署模式
        if not hasattr(self, 'conv'):
            # 定义新的3x3卷积层用于部署
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        # 获取等效的卷积核和偏置
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias

    def get_equivalent_kernel_bias(self):
        # 获取等效的卷积核和偏置
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        # 返回合并后的卷积核和偏置
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        # 将1x1卷积核填充为3x3
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        # 融合批归一化参数
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        # 返回融合后的卷积核和偏置
        return kernel * t, beta - running_mean * gamma / std

if __name__ == '__main__':
    model = RepVggBlock(ch_in=32, ch_out=32)
    input = torch.randn(1, 32, 50, 50)
    # 训练时
    output = model(input)
    print('input_size:', input.size())
    print('output_size:', output.size())

    # 推理时 ==>转换为部署模式
    model.convert_to_deploy()
    output_deploy = model(input)
    print('output_size after deploy:', output_deploy.size())
    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")