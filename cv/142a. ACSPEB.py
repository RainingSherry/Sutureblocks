import torch
import torch.nn as nn
import torch.nn.functional as F

class ACSPEB(nn.Module):
    """
    ACSPEB: Adaptive Channel-Spatial Perception Enhanced Block
    自适应通道-空间感知增强模块
    专为 Mamba 架构设计，通过双维度互补注意力、通道自适应重组与门控残差，解决特征碎片化与一维扫描缺陷。
    """
    def __init__(self, in_channels, reduction=4):
        super(ACSPEB, self).__init__()
        
        # 中间降维通道数，保证模块的轻量化
        hidden_channels = max(in_channels // reduction, 16)

        # ================== 创新点 2：通道-空间双维度互补注意力 ==================
        # 通道注意力提取分支 (融合 GAP 与 GMP)
        self.ca_mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=False)
        )
        
        # 空间注意力提取分支 (局部上下文感知)
        self.sa_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

        # ================== 创新点 3：特征校准与门控残差融合机制 ==================
        # 1x1 卷积特征校准模块，消除通道置换带来的特征碎片化问题
        self.calibration = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True)
        )
        
        # 分层门控残差滤波器，用于屏蔽冗余噪声
        self.gated_filter = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        identity = x

        # ================== 步骤 1：双维度互补注意力建模 ==================
        # 提取通道级全局依赖
        gap = F.adaptive_avg_pool2d(x, 1)
        gmp = F.adaptive_max_pool2d(x, 1)
        # 将平均池化与最大池化的结果聚合，生成通道注意力权重
        ca_weight = torch.sigmoid(self.ca_mlp(gap) + self.ca_mlp(gmp))  # [B, C, 1, 1]

        # 提取空间级局部上下文
        sa_avg = torch.mean(x, dim=1, keepdim=True)
        sa_max, _ = torch.max(x, dim=1, keepdim=True)
        # 拼接空间统计量，生成空间注意力权重
        sa_weight = self.sa_conv(torch.cat([sa_avg, sa_max], dim=1))    # [B, 1, H, W]

        # 双维度特征同步增强
        x_enhanced = x * ca_weight * sa_weight

        # ================== 步骤 2：注意力引导的自适应通道置换 ==================
        # 创新点 1：利用通道注意力权重对特征进行降序排序
        # 提取权重大小的索引，自适应聚焦高价值通道，消除随机 Shuffle 的盲目性
        sorted_indices = torch.argsort(ca_weight, dim=1, descending=True) # [B, C, 1, 1]
        
        # 扩展索引至空间维度以便进行 Gather 操作
        gather_indices = sorted_indices.expand(B, C, H, W)
        
        # 根据高价值通道优先级对增强后的特征进行物理重组
        x_permuted = torch.gather(x_enhanced, dim=1, index=gather_indices)

        # ================== 步骤 3：特征校准与门控残差融合 ==================
        # 对重组后的特征进行 1x1 卷积校准，恢复通道间的平滑映射关系
        x_calib = self.calibration(x_permuted)

        # 生成门控滤波掩码，对高频特征噪声进行抑制
        gate = self.gated_filter(x_calib)
        
        # 将清洗并增强后的特征通过门控机制融合至恒等映射中
        out = identity + x_calib * gate

        return out


# 使用示例
if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    input_tensor = torch.randn(1, 64, 256, 256).to(device)
    
    model = ACSPEB(in_channels=64).to(device)
    
    print(model)
    
    output_tensor = model(input_tensor)
    
    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape)
    
    print("\n哔哩哔哩/微信公众号: CV缝合救星,独家整理! \n")