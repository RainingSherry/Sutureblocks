import torch
import torch.nn as nn

"""
    通过线性层提取水平与垂直维度上的特征（双分支特征增强）
        写作思路与代码讲解：https://www.bilibili.com/video/BV1RLfPYsEzG/
        作用位置：通过独立的高度方向投影和宽度方向投影实现空间维度（H、W）建模，在特征输入之前加入。
        主要功能：①捕捉图像H、W方向的长依赖关系。②垂直/水平维度的全局上下文信息
        代码使用方式与写作思路请务必看视频~
"""

class HeightWidthFeatureMLP(nn.Module):
    def __init__(self, image_height=224, image_width=224, channel_count=3):
        # 输入参数为图像的高度、宽度以及通道数
        super().__init__()
        # 定义沿高度方向的线性变换层
        self.height_projection = nn.Linear(image_height, image_height)
        # 定义沿宽度方向的线性变换层
        self.width_projection = nn.Linear(image_width, image_width)
        # 定义融合层，用于融合来自不同方向的信息
        self.feature_fusion = nn.Linear(3 * channel_count, channel_count)

        self.convolution_batchnorm_relu = nn.Sequential(
            # 添加2D卷积层，输入通道数是原通道数的3倍，输出通道数为channel_count
            nn.Conv2d(3 * channel_count, channel_count, kernel_size=1, stride=1),
            # 批量归一化层，对channel_count个特征图进行归一化
            nn.BatchNorm2d(channel_count),
            # ReLU激活函数，inplace=True表示直接在输入数据上进行修改以节省内存
            nn.ReLU(inplace=True),
        )

    def forward(self, input_tensor):
        # 保留原始输入作为残差连接
        original_input = input_tensor

        # 因为 nn.Linear 默认对最后一个维度进行操作，沿高度方向进行线性变换，并调整维度顺序
        # [B,C,H,W]  ---》 [B,C,W,H]
        height_transformed = self.height_projection(input_tensor.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        # 沿宽度方向进行线性变换
        width_transformed = self.width_projection(input_tensor)

        """
            这里可以再加一些自注意力
        """

        # 在通道维度上合并不同的特征
        merged_features = torch.cat([height_transformed, width_transformed, original_input], dim=1)

        # 维度还原方式① https://www.bilibili.com/video/BV1vfZnYTEE2/
        total_merged_features = merged_features.permute(0, 2, 3, 1)
        # 同上：因为 nn.Linear 默认对最后一个维度进行操作，所以这里先线性层后调换位置【常见套路】
        output_tensor = self.feature_fusion(total_merged_features).permute(0, 3, 1, 2)

        # 维度还原方式②
        # output_tensor = self.convolution_batchnorm_relu(merged_features)
        return output_tensor

if __name__ == '__main__':
    input_tensor = torch.randn(1, 32, 224, 224)
    print(input_tensor.shape)
    height_width_mlp = HeightWidthFeatureMLP(image_height=224, image_width=224, channel_count=32)
    output = height_width_mlp(input_tensor)
    print(f'Input size: {input_tensor.size()}')
    print(f'Output size: {output.size()}')
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：微创新·代码无误")