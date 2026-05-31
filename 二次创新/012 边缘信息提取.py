import torch
import torch.nn as nn

"""
    简单实现特征图边缘感知功能：
        写作思路与代码讲解：https://www.bilibili.com/video/BV1Gd3azsEuZ
        作用位置：任何单一输出特征后，或者任何即插即用模块中，关注图像边缘和细节信息，同时保留整体特征。
        主要功能：1、通过平均池化与原始输入的差分提取边缘信息。2、利用这些边缘信息生成权重图。3、通过残差连接的方式增强原始特征。
        代码使用方式与写作思路请务必看视频~  
"""

class EdgeAwareFeatureEnhancer(nn.Module):
    def __init__(self, in_channels):
        super(EdgeAwareFeatureEnhancer, self).__init__()
        # 边缘提取模块 - 使用3x3平均池化
        self.edge_extractor = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        # 权重生成网络 - 1x1卷积 + BN + Sigmoid
        self.weight_generator = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        # 边缘特征提取
        edge_features = x - self.edge_extractor(x)
        # 生成边缘感知权重
        edge_weights = self.weight_generator(edge_features)
        # 加权残差连接
        enhanced_features = edge_weights * x + x
        return enhanced_features

if __name__ == "__main__":
    # 测试代码
    x = torch.randn(1, 32, 128, 128)
    enhancer = EdgeAwareFeatureEnhancer(in_channels=32)
    output_tensor = enhancer(x)
    print(f'输入特征尺寸: {x.size()}')
    print(f'输出特征尺寸: {output_tensor.size()}')
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：微创新·代码无误")
