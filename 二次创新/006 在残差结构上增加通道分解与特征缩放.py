import torch
import torch.nn as nn

"""
    在残差结构上增加通道分解（功能）与特征缩放（加权方式）：
        写作思路与代码讲解：https://www.bilibili.com/video/BV1E8jwzbE2F/
        作用位置：残差结构通用。
        主要功能：①通道分解：得到差异特征。②特征缩放：防止梯度爆炸（nan  难！）。
"""

class FeatureScaler(nn.Module):
    def __init__(self, embedding_dims, initial_value=0.1, gradient_enabled=True):
        super(FeatureScaler, self).__init__()
        # 定义一个可学习的参数scaling_factor，其初始值为initial_value乘以全1张量
        # 张量的形状为(1, embedding_dims, 1, 1)，是否可训练由gradient_enabled决定
        self.scaling_factor = nn.Parameter(
            initial_value * torch.ones((1, embedding_dims, 1, 1)),
            requires_grad=gradient_enabled
        )
    def forward(self, input_tensor):
        # 将输入张量与缩放因子相乘并返回结果
        return input_tensor * self.scaling_factor


class Aggregation(nn.Module):
    def __init__(self, embedding_dims):
        super(Aggregation, self).__init__()
        self.embedding_dims = embedding_dims
        self.feedforward_channels = self.embedding_dims

        # 定义一个1x1的卷积层，用于通道分解
        # 输入通道数为前馈通道数，输出通道数为1
        self.channel_decomposer = nn.Conv2d(
            in_channels=self.feedforward_channels,
            out_channels=1,
            kernel_size=1
        )

        # 定义一个FeatureScaler层，用于对残差进行缩放
        self.scaling_layer = FeatureScaler(
            self.feedforward_channels,
            initial_value=1e-5,
            gradient_enabled=True
        )
        # 定义GELU激活函数，用于分解后的激活操作
        self.activation = nn.GELU()

    def forward(self, input_tensor):
        """
            图像特征通道聚合
        """
        # 使用通道分解卷积层对输入张量进行分解
        decomposed = self.channel_decomposer(input_tensor)
        # 对分解后的结果应用GELU激活函数
        decomposed = self.activation(decomposed)
        # 计算残差，即输入张量减去分解后的结果，得到差异特征
        residual = input_tensor - decomposed

        """
              图像特征缩放
        """
        # 使用缩放层对残差进行缩放
        scaled_residual = self.scaling_layer(residual)

        # 将输入张量与缩放后的残差相加得到输出
        output = input_tensor + scaled_residual
        return output

if __name__ == '__main__':
    input_tensor = torch.randn(1, 32, 50, 50)
    model = Aggregation(32)
    output = model(input_tensor)
    print(' input_size:', input_tensor.size())
    print(' output_size:', output.size())
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：微创新·代码无误")