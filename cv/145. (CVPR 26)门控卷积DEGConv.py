import torch
import torch.nn as nn
import torch.nn.functional as F

class DEGConv(nn.Module):
    """
    DEGConv 模块（方向引导边缘门控卷积）
    参考论文里的 DEGConv 设计，用于增强裂缝边缘方向敏感性,
    加强纹理细节提取，同时保持轻量级计算。

    核心思想：
    1) 方向引导边缘卷积，从输入特征计算方向信息
    2) 条状卷积提取水平和垂直边缘响应
    3) 门控机制动态控制不同方向信息的流动
    """

    def __init__(self, in_channels, out_channels, reduction=4):
        super(DEGConv, self).__init__()
        # 1x1 压缩通道（CV缝合救星轻量化 trick）
        self.shrink = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.act = nn.ReLU(inplace=True)

        # 条状卷积：水平和垂直方向滤波
        self.conv_h = nn.Conv2d(in_channels // reduction, in_channels // reduction,
                                kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.conv_v = nn.Conv2d(in_channels // reduction, in_channels // reduction,
                                kernel_size=(3, 1), padding=(1, 0), bias=False)

        # 门控分支，用于融合提取到的方向信息
        self.gate = nn.Conv2d(in_channels // reduction, in_channels // reduction, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        # 拼接后再恢复到 out_channels
        self.expand = nn.Conv2d((in_channels // reduction) * 2, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        # 压缩通道
        x_shrink = self.act(self.shrink(x))

        # 水平边缘响应
        feat_h = self.act(self.conv_h(x_shrink))
        # 垂直边缘响应
        feat_v = self.act(self.conv_v(x_shrink))

        # 门控权重
        gate = self.sigmoid(self.gate(x_shrink))

        # 门控融合
        h_gated = feat_h * gate
        v_gated = feat_v * gate

        # 通道拼接
        combined = torch.cat([h_gated, v_gated], dim=1)

        # 恢复通道
        out = self.expand(combined)
        return out


# --------------------- 测试 DEGConv 模块 ---------------------
if __name__ == "__main__":
    # CV 缝合救星提示: 大多数视觉模型输入尺寸是 (batch, channels, height, width)
    dummy_input = torch.randn(2, 64, 256, 256)  # 例如 batch=2, 64 通道, 256x256 图

    # 初始化 DEGConv: 64 --> 128 通道变换
    degconv = DEGConv(in_channels=64, out_channels=64)

    print("=== CV 缝合救星 | DEGConv 模块结构 ===")
    print(degconv)

    output = degconv(dummy_input)
    print("\n=== 输入形状 ===", dummy_input.shape)
    print("=== 输出形状 ===", output.shape)