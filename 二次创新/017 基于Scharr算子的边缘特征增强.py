import torch
import torch.nn as nn

"""
    基于Scharr算子的边缘特征增强模块：
        写作思路与代码讲解：https://www.bilibili.com/video/BV1oQtjzfELt/
        作用位置：任何单一输出特征后，或者任何即插即用模块中。
        主要功能：通过Scharr边缘检测算子提取特征图边缘信息，并生成注意力权重来增强原始特征。
        代码层面：①Scharr卷积层：使用3×3 Scharr算子计算水平和垂直方向边缘。
                ②边缘强度计算：通过L2范数融合双方向边缘响应。
                ③归一化与激活：对边缘图进行批归一化（BN）和GELU激活。
                ④注意力机制：将归一化的边缘图作为注意力掩码与输入特征相乘。
"""

def create_norm_layer(norm_config, num_features):
    """创建归一化层，目前仅支持Batch Normalization"""
    norm_type = norm_config.get('type', 'BN')
    requires_grad = norm_config.get('requires_grad', True)

    if norm_type == 'BN':
        norm_layer = nn.BatchNorm2d(num_features)
    else:
        raise NotImplementedError(f"不支持的归一化类型: {norm_type}")
    # 设置是否需要梯度更新
    for param in norm_layer.parameters():
        param.requires_grad = requires_grad
    return norm_type, norm_layer

class ScharrEdgeEnhancement(nn.Module):
    """基于Scharr算子的边缘增强模块"""
    def __init__(self, in_channels):
        super().__init__()

        # 配置归一化层参数
        norm_config = dict(type='BN', requires_grad=True)

        # 定义Scharr算子的x和y方向卷积核
        scharr_kernel_x = torch.tensor(
            [[-3., 0., 3.],
             [-10., 0., 10.],
             [-3., 0., 3.]]
        ).unsqueeze(0).unsqueeze(0)  # 扩展为[1, 1, 3, 3]

        scharr_kernel_y = torch.tensor(
            [[-3., -10., -3.],
             [0., 0., 0.],
             [3., 10., 3.]]
        ).unsqueeze(0).unsqueeze(0)  # 扩展为[1, 1, 3, 3]

        # 创建深度可分离卷积层（分组卷积实现）
        self.conv_x = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, padding=1,
            groups=in_channels, bias=False
        )

        self.conv_y = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, padding=1,
            groups=in_channels, bias=False
        )

        # 初始化卷积核权重（固定为Scharr算子）
        self.conv_x.weight.data = scharr_kernel_x.repeat(in_channels, 1, 1, 1)
        self.conv_y.weight.data = scharr_kernel_y.repeat(in_channels, 1, 1, 1)

        # 初始化归一化层和激活函数
        self.norm = create_norm_layer(norm_config, in_channels)[1]
        self.activation = nn.GELU()

    def forward(self, x):
        # 计算x和y方向的边缘响应
        edge_response_x = self.conv_x(x)
        edge_response_y = self.conv_y(x)

        # 计算边缘强度（L2范数）
        edge_strength = torch.sqrt(edge_response_x ** 2 + edge_response_y ** 2)
        # 生成边缘注意力权重
        edge_attention = self.activation(self.norm(edge_strength))
        # 应用注意力机制增强特征
        enhanced_feature = x * edge_attention

        return enhanced_feature

if __name__ == '__main__':
    x = torch.randn(1, 32, 50, 50)
    model = ScharrEdgeEnhancement(in_channels=32)
    output = model(x)
    print(f"输入张量形状: {x.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：微创新·代码无误")