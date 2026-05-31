import torch
import torch.nn as nn

""" 
   基于窗口切分的区域级特征重标定机制：
        写作思路与代码讲解：https://www.bilibili.com/video/BV17PDfBpEQR/
        作用位置：任何单一特征处理时/任何普通卷积，或者任何即插即用模块中。
        主要功能（写作要点）：①突出局部显著区域中的关键响应；
                            ②增强对边界、纹理、细长结构及不规则区域的刻画能力；
                            ③保留不同区域之间的空间差异性。（将在本视频的写作部分展开阐述）
        代码层面：先将输入特征图按空间区域划分为四个局部子块，再对每个子块分别施加通道注意力与空间注意力，从而实现局部区域的细粒度增强。
"""
# Window-based Regional Feature Recalibration WRFR
class ChannelAttention(nn.Module):
    """
    通道注意力模块
    输入:  [B, C, H, W]
    输出:  [B, C, 1, 1]
    """
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()

        hidden_channels = max(1, in_channels // reduction)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_weight = self.mlp(self.avg_pool(x))
        max_weight = self.mlp(self.max_pool(x))
        attention = self.sigmoid(avg_weight + max_weight)
        return attention


class SpatialAttention(nn.Module):
    """
    空间注意力模块
    输入:  [B, C, H, W]
    输出:  [B, 1, H, W]
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), "kernel_size 只能为 3 或 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        pooled = torch.cat([avg_map, max_map], dim=1)
        attention = self.sigmoid(self.conv(pooled))
        return attention


class WindowRegionalFeatureRecalibration(nn.Module):
    """
    基于窗口切分的区域级特征重标定模块
    """
    def __init__(self, in_channels, reduction=16, spatial_kernel=7):
        super(WindowRegionalFeatureRecalibration, self).__init__()

        self.ca = ChannelAttention(in_channels, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, feature_map):
        """
        feature_map: [B, C, H, W]
        """
        if feature_map.dim() != 4:
            raise ValueError(f"输入张量必须为 4 维 [B, C, H, W]，但当前维度为 {feature_map.dim()}")

        b, c, h, w = feature_map.shape

        if h < 2 or w < 2:
            raise ValueError(f"输入特征图的高和宽必须都 >= 2，当前输入尺寸为 ({h}, {w})")

        # 沿高度方向切分为上下两部分
        top_half, bottom_half = feature_map.chunk(2, dim=2)

        # 上半部分沿宽度方向切分为左上、右上
        top_left, top_right = top_half.chunk(2, dim=3)

        # 下半部分沿宽度方向切分为左下、右下
        bottom_left, bottom_right = bottom_half.chunk(2, dim=3)

        # 左上区域：通道重标定 + 空间重标定
        top_left = self.ca(top_left) * top_left
        top_left = self.sa(top_left) * top_left

        # 右上区域：通道重标定 + 空间重标定
        top_right = self.ca(top_right) * top_right
        top_right = self.sa(top_right) * top_right

        # 左下区域：通道重标定 + 空间重标定
        bottom_left = self.ca(bottom_left) * bottom_left
        bottom_left = self.sa(bottom_left) * bottom_left

        # 右下区域：通道重标定 + 空间重标定
        bottom_right = self.ca(bottom_right) * bottom_right
        bottom_right = self.sa(bottom_right) * bottom_right

        # 恢复上半部分
        top_fused = torch.cat([top_left, top_right], dim=3)

        # 恢复下半部分
        bottom_fused = torch.cat([bottom_left, bottom_right], dim=3)

        # 恢复完整特征图
        enhanced_feature_map = torch.cat([top_fused, bottom_fused], dim=2)

        return enhanced_feature_map

if __name__ == '__main__':
    x = torch.rand(1, 32, 50, 50)
    model = WindowRegionalFeatureRecalibration(in_channels=32)
    output_tensor = model(x)
    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {output_tensor.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：微创新·代码无误")