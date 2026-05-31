import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelShuffle(nn.Module):
    """
    通道混洗（Channel Shuffle）实现
    用于打破分组卷积带来的通道信息孤岛
    """
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, self.groups, channels // self.groups, height, width)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, channels, height, width)
        return x

class SpectralGatedUnit(nn.Module):
    """
    频域门控单元 (Spectral Gated Unit)
    利用频域全局先验生成动态门控信号
    """
    def __init__(self, channels):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.gate(self.gap(x))

class S_GKA(nn.Module):
    """
    S-GKA: Spectral-Gated Kernels Attention
    频域门控核注意力模块
    专注于医学影像中低对比度目标的动态选择性增强
    """
    def __init__(self, in_channels=3, mid_channels=32, groups=4):
        """
        参数:
            in_channels: 输入通道数
            mid_channels: 隐藏层维度
            groups: 通道混洗分组
        """
        super().__init__()
        
        # 初始特征投影
        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # 1. 异构空间路径 (多尺度 DWC)
        self.dwc3 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1, groups=mid_channels, bias=False)
        self.dwc5 = nn.Conv2d(mid_channels, mid_channels, 5, padding=2, groups=mid_channels, bias=False)
        
        # 2. 频域引导路径
        self.spec_gate = SpectralGatedUnit(mid_channels)
        
        # 3. 动态核选择门控
        self.kernel_selection = nn.Sequential(
            nn.Conv2d(mid_channels, 2, 1, bias=False),
            nn.Softmax(dim=1)
        )

        # 4. 后处理与重组
        self.channel_shuffle = ChannelShuffle(groups=groups)
        self.proj = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1, bias=False)
        )

    def forward(self, x):
        # 映射至中间特征空间
        identity = x
        feat = self.pre_conv(x)
        
        # 频域门控生成
        g = self.spec_gate(feat)
        
        # 异构多尺度特征提取
        x3 = self.dwc3(feat)
        x5 = self.dwc5(feat)
        
        # 频域信号引导的空间响应
        x3_g = x3 * g
        x5_g = x5 * g
        
        # 动态核权重分配 (基于内容决定依赖哪个尺度的核)
        w = self.kernel_selection(x3_g + x5_g)
        w3, w5 = w[:, 0:1, :, :], w[:, 1:2, :, :]
        
        # 选择性融合
        fused = x3_g * w3 + x5_g * w5
        
        # 通道混洗交互
        out = self.channel_shuffle(fused)
        out = self.proj(out)
        
        # 残差闭环
        return out + identity

# 使用示例
if __name__ == "__main__":
    # 配置运行设备
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 模拟医学影像输入: [Batch, Channel, Height, Width]
    input_tensor = torch.randn(1, 3, 256, 256).to(device)

    # 实例化 S-GKA 模块
    model = S_GKA(in_channels=3, mid_channels=32, groups=4).to(device)

    print(model)

    # 前向传播测试
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)  
    print("output_tensor_shape :", output_tensor.shape)

    print("\n哔哩哔哩/微信公众号: CV缝合救星, 独家整理! \n")