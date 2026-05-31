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

class FrequencyGate(nn.Module):
    """
    频域全局引导门控 (Frequency-Domain Global Gating)
    通过DCT捕捉全局纹理统计量，生成特征选择门控
    """
    def __init__(self, channels):
        super().__init__()
        # 利用GAP简化DCT的频率分量捕捉逻辑
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 模拟频域全局先验对空域特征的重加权
        return x * self.gate(x)

class PFESA(nn.Module):
    """
    PFESA: Progressive Frequency-Edge Selective Attention
    渐进式频域-边缘选择性注意力模块
    针对医学影像中的多尺度病灶与复杂边界设计的魔改架构
    """
    def __init__(self, in_channels=3, groups=4, dilated_rate=2):
        """
        参数:
            in_channels: 输入通道数（主函数测试输入为3）
            groups: 分组数
            dilated_rate: 扩张卷积率
        """
        super().__init__()
        # 初始通道映射（处理输入通道极小的情况）
        self.mid_channels = 32 if in_channels < 32 else in_channels
        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.mid_channels, 1, bias=False),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(inplace=True)
        )

        # 1. 异构空间路径 (Heterogeneous Spatial Paths)
        self.branch_gpc = nn.Conv2d(self.mid_channels, self.mid_channels, 1, groups=groups, bias=False)
        self.branch_dwc3 = nn.Conv2d(self.mid_channels, self.mid_channels, 3, padding=1, groups=self.mid_channels, bias=False)
        self.branch_dwc5 = nn.Conv2d(self.mid_channels, self.mid_channels, 5, padding=2, groups=self.mid_channels, bias=False)
        self.branch_ddc = nn.Conv2d(self.mid_channels, self.mid_channels, 3, padding=dilated_rate, dilation=dilated_rate, groups=self.mid_channels, bias=False)

        # 2. 频域引导与通道交互
        self.freq_gate = FrequencyGate(self.mid_channels)
        self.channel_shuffle = ChannelShuffle(groups=groups)
        
        # 3. 渐进式精炼与投影 (Progressive Refinement)
        self.refine = nn.Sequential(
            nn.Conv2d(self.mid_channels, self.mid_channels, 3, padding=1, groups=self.mid_channels, bias=False),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_channels, in_channels, 1, bias=False) # 映射回原始通道数
        )

    def forward(self, x):
        # 输入投影
        feat = self.pre_conv(x)
        
        # 多尺度异构并行提取
        out_gpc = self.branch_gpc(feat)
        out_dwc3 = self.branch_dwc3(feat)
        out_dwc5 = self.branch_dwc5(feat)
        out_ddc = self.branch_ddc(feat)
        
        # 逐元素选择性相加
        spatial_fusion = out_gpc + out_dwc3 + out_dwc5 + out_ddc
        
        # 频域全局特征引导
        guided_feat = self.freq_gate(spatial_fusion)
        
        # 通道混洗打破信息孤岛
        shuffled_feat = self.channel_shuffle(guided_feat)
        
        # 渐进式特征投影输出
        out = self.refine(shuffled_feat)
        
        # 残差连接
        return out + x

# 使用示例
if __name__ == "__main__":
    
    # 配置运行设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 模拟医学图像输入: [Batch, Channel, Height, Width]
    input_tensor = torch.randn(1, 3, 256, 256).to(device)
    
    # 实例化 PFESA 模块
    model = PFESA(in_channels=3, groups=1, dilated_rate=2).to(device) # 输入通道为3时groups设为1
    
    print(model)
    
    # 前向传播
    output_tensor = model(input_tensor)
    
    # 打印维度验证算子正确性
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape)
    
    print("\n哔哩哔哩/微信公众号: CV缝合救星, 独家整理! \n")