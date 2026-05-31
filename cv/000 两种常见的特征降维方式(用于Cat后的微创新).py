import torch
import torch.nn as nn

"""
    两种常见的特征降维改进方式（用于Cat后的微创新）
        写作思路与代码讲解：https://www.bilibili.com/video/BV1vfZnYTEE2/
        作用位置：在通道拼接以后，通道会变为原来的C倍，我们想将它变为原始的大小。
        主要功能：对输入特征图进行通道数的降维操作。
        代码使用方式与写作思路请务必看视频~
"""

class ChannelReducer(nn.Module):
    def __init__(self, channel_num):
        super().__init__()
        """
            注意：根据降维的程度进行修改
        """
        self.feature_fuser = nn.Linear(3 * channel_num, channel_num)  # 定义融合层，用于融合来自不同方向的信息

        self.conv_bn_activation = nn.Sequential(
            # 添加2D卷积层，输入通道数是原通道数的3倍，输出通道数为channel_num
            nn.Conv2d(3 * channel_num, channel_num, kernel_size=1, stride=1),
            # 批量归一化层，对channel_num个特征图进行归一化
            nn.BatchNorm2d(channel_num),
            # ReLU激活函数，inplace=True表示直接在输入数据上进行修改以节省内存
            nn.ReLU(inplace=True),
        )

    def forward(self, x1,x2,x3):
        # 在通道维度上合并不同的特征
        merged_features = torch.cat([x1, x2, x3], dim=1)

        # 维度还原方式① 通过线性变换层进行维度变换，nn.Linear 默认对最后一个维度进行操
        # 调换位置 B C H W === B H W C
        # transposed_features = merged_features.permute(0, 2, 3, 1)
        # output_tensor = self.feature_fuser(transposed_features)
        # output_tensor = output_tensor.permute(0, 3, 1, 2)

        # 维度还原方式②
        output_tensor = self.conv_bn_activation(merged_features)
        return output_tensor

if __name__ == '__main__':
    x1 = torch.randn(1, 64, 50, 50)
    x2 = torch.randn(1, 64, 50, 50)
    x3 = torch.randn(1, 64, 50, 50)
    channel_reducer = ChannelReducer(channel_num=64)
    output = channel_reducer(x1,x2,x3)
    print(f'Input size: {x1.size()}')
    print(f'Output size: {output.size()}')
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：微创新·代码无误")