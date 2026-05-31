import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Module: Dynamic Gradient-aware Routing Module (DGRM)
# 动态梯度感知路由模块
# 
# Abstract:
# 本模块旨在解决传统多尺度注意力机制中存在的特征降维瓶颈与高频边界模糊问题。
# 通过引入零降维的局部信道交互、基于差分算子的梯度空间调制，以及数据驱动的
# 动态感受野软路由，DGRM 能够在保持极低计算开销的同时，显著增强网络对复杂视觉
# 图像拓扑结构的细粒度表征能力。
# =============================================================================

class LocalCrossChannelAttention(nn.Module):
    """
    局部跨信道注意力机制 (Local Cross-Channel Attention, LCCA).
    
    设计动机：
    传统的通道注意力（如 SE-Net）通常采用带有缩放因子 (Reduction Ratio) 的全连接层，
    这种瓶颈结构 (Bottleneck) 会不可避免地导致通道信息的硬性丢失。
    LCCA 通过一维卷积在通道维度上直接建立局部邻域依赖，实现了零降维 (Zero-Reduction)
    的跨通道信息交互，且计算复杂度与通道数呈严格线性关系。
    """
    def __init__(self, channels, kernel_size=3):
        super(LocalCrossChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 采用无偏置的一维卷积进行通道特征平滑与局部交互建模
        self.conv1d = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape [B, C, H, W]
        Returns:
            Tensor: Channel-attention modulated tensor of shape [B, C, H, W]
        """
        # 1. 提取全局空间分布的低阶 (均值) 与高阶 (峰值) 统计量
        y_avg = self.avg_pool(x).squeeze(-1).transpose(-1, -2)  # Shape: [B, 1, C]
        y_max = self.max_pool(x).squeeze(-1).transpose(-1, -2)  # Shape: [B, 1, C]
        
        # 2. 沿通道维度进行 1D 卷积操作，捕获局部信道关联
        attn_avg = self.conv1d(y_avg).transpose(-1, -2).unsqueeze(-1) # Shape: [B, C, 1, 1]
        attn_max = self.conv1d(y_max).transpose(-1, -2).unsqueeze(-1) # Shape: [B, C, 1, 1]
        
        # 3. 统计量融合与非线性特征激活
        attn = self.sigmoid(attn_avg + attn_max)
        
        return x * attn


class GradientAwareSpatialAttention(nn.Module):
    """
    梯度先验空间注意力机制 (Gradient-Aware Spatial Attention, GASA).
    
    设计动机：
    常规的空间注意力过度依赖全局池化，容易引起高频边界（如病灶轮廓、微小血管）的平滑与衰减。
    GASA 引入了一阶空间差分算子 (Spatial Difference Operator)，显式提取特征图的水平与垂直梯度，
    为空间注意力权重分布注入了极具判别力的边界归纳偏置 (Boundary Inductive Bias)。
    """
    def __init__(self):
        super(GradientAwareSpatialAttention, self).__init__()
        # 利用 7x7 大感受野卷积整合统计先验与梯度先验
        self.fusion_conv = nn.Conv2d(4, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape [B, C, H, W]
        Returns:
            Tensor: Spatial-attention modulated tensor of shape [B, C, H, W]
        """
        # 1. 基础空间统计量特征提取 (Mean & Max)
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Shape: [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True) # Shape: [B, 1, H, W]
        
        # 2. 空间梯度计算 (Spatial Gradient Estimation)
        # 通过相邻像素差分近似计算水平 (X轴) 与垂直 (Y轴) 的高频特征梯度
        grad_x = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
        grad_x = F.pad(grad_x, (0, 1, 0, 0))
        grad_x_out = torch.mean(grad_x, dim=1, keepdim=True) # Shape: [B, 1, H, W]
        
        grad_y = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
        grad_y = F.pad(grad_y, (0, 0, 0, 1))
        grad_y_out = torch.mean(grad_y, dim=1, keepdim=True) # Shape: [B, 1, H, W]
        
        # 3. 跨模态先验融合与权重生成
        spatial_prior = torch.cat([avg_out, max_out, grad_x_out, grad_y_out], dim=1) # Shape: [B, 4, H, W]
        attn = self.sigmoid(self.fusion_conv(spatial_prior))
        
        return x * attn


class AdaptiveReceptiveRouting(nn.Module):
    """
    自适应感受野路由机制 (Adaptive Receptive-field Routing, ARR).
    
    设计动机：
    针对图像中目标尺度差异巨大的问题，ARR 采用选择性核 (Selective Kernel) 范式。
    通过提取全局上下文向量，模型能够动态预测不同空洞率 (Dilation=1, 2, 3) 卷积分支的
    重要性分布。这种软路由 (Soft-routing) 策略有效替代了传统的静态特征拼接 (Concatenation)，
    实现了特征感受野与目标尺度的自适应对齐。
    """
    def __init__(self, in_channels, out_channels):
        super(AdaptiveReceptiveRouting, self).__init__()
        
        # 特征维度映射
        self.align = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        # 构建具有不同膨胀率的并行特征提取分支 (模拟局部、中等与全局感受野)
        self.branch_d1 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1, dilation=1), nn.ReLU(inplace=True))
        self.branch_d2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=2, dilation=2), nn.ReLU(inplace=True))
        self.branch_d3 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=3, dilation=3), nn.ReLU(inplace=True))
        
        # 动态权重生成网络 (Dynamic Weight Generation Network)
        self.gap = nn.AdaptiveAvgPool2d(1)
        mid_channels = max(8, out_channels // 4)
        
        # 为了保障模型在极小批量 (如 Batch Size = 1) 推理时的数值稳定性，
        # 采用标准全连接范式 (bias=True)，彻底消除对 Batch Normalization 方差计算的依赖。
        self.fc1 = nn.Conv2d(out_channels, mid_channels, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(mid_channels, 3 * out_channels, 1, bias=True)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape [B, C_in, H, W]
        Returns:
            Tensor: Multi-scale routed tensor of shape [B, C_out, H, W]
        """
        x = self.align(x)
        b, c, h, w = x.size()
        
        # 1. 并行提取多尺度感受野特征
        feat_1 = self.branch_d1(x)
        feat_2 = self.branch_d2(x)
        feat_3 = self.branch_d3(x)
        
        # 2. 全局上下文聚合
        U = feat_1 + feat_2 + feat_3
        
        # 3. 计算通道级的动态路由门控系数
        s = self.gap(U)
        z = self.relu(self.fc1(s))
        route_weights = self.fc2(z).view(b, 3, c, 1, 1) # Shape: [B, 3, C, 1, 1]
        
        # 4. 跨分支 Softmax 归一化，确保特征贡献度总和为 1
        route_weights = F.softmax(route_weights, dim=1)
        
        # 5. 执行自适应软融合 (Adaptive Soft-Fusion)
        out = (feat_1 * route_weights[:, 0, :, :, :] + 
               feat_2 * route_weights[:, 1, :, :, :] + 
               feat_3 * route_weights[:, 2, :, :, :])
               
        return out


class DGRM(nn.Module):
    """
    动态梯度感知路由模块 (Dynamic Gradient-aware Routing Module).
    
    架构逻辑：
    该模块以串行方式依次集成局部跨信道交互、梯度空间调制与自适应多尺度路由。
    阶段 I: 通道维度协同与去冗余
    阶段 II: 空间维度高频边界强化
    阶段 III: 多尺度感受野自适应对齐
    """
    def __init__(self, in_channels, out_channels):
        super(DGRM, self).__init__()
        
        self.lcca = LocalCrossChannelAttention(in_channels)
        self.gasa = GradientAwareSpatialAttention()
        self.arr = AdaptiveReceptiveRouting(in_channels, out_channels)

    def forward(self, x):
        # 串行特征调制计算图
        x = self.lcca(x)
        x = self.gasa(x)
        out = self.arr(x)
        
        return out


# 使用示例
if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 构造标准输入特征映射
    input_tensor = torch.randn(1, 3, 256, 256).to(device)
    
    # 实例化 DGRM 网络模块
    model = DGRM(in_channels=3, out_channels=3).to(device)
    
    # 输出模型层级结构参数
    print(model)
    
    # 执行前向推演计算
    output_tensor = model(input_tensor)
    
    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape) 
    
    print("\n哔哩哔哩/微信公众号: CV缝合救星,独家整理! \n")