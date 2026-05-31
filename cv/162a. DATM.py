import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Module: Dynamic Adaptive Triangular-Masked Convolution (DATM-Conv)
# 动态自适应三角掩码卷积模块
# 
# Abstract:
# 本模块旨在解决原生单尺度三角掩码卷积在复杂真实图像去噪中存在的感受野局限问题。
# 通过结合“多尺度空洞三角掩码（Multi-scale Dilated Triangular Mask）”与
# “自适应上下文路由（Adaptive Context Routing）”，DATM 模块能够在严格恪守
# 上三角有向无环图（DAG）盲点约束的前提下，动态感知输入特征的频域响应，为不同
# 尺度的结构纹理自动分配最优的掩码感受野权重，极大地提升了模型在复杂背景下对
# 孤立噪声的解耦能力和细节保持度。
# =============================================================================

class Dilated_TM_Conv(nn.Conv2d):
    """
    底层算子：空洞三角掩码卷积 (Dilated Triangular-Masked Convolution)
    
    设计动机：
    为 TM-Conv 引入空洞率（Dilation），在不增加参数量的前提下扩大感受野，
    同时通过严格的上三角屏蔽约束，确保即使感受野扩大，依然不会破坏下游
    特征平移操作所需的盲点几何拓扑。
    """
    def __init__(self, in_channels: int, out_channels: int, k: int, dilation: int = 1):
        # 动态计算 padding，确保各类空洞率下的特征图空间分辨率对齐
        padding = (k - 1) * dilation // 2
        super().__init__(in_channels, out_channels, kernel_size=k,
                         stride=1, padding=padding, dilation=dilation, groups=1, bias=False)

        # 构建并固化上三角约束矩阵
        mask = torch.ones_like(self.weight)
        tri2d = torch.triu(torch.ones(k, k, dtype=mask.dtype, device=mask.device))
        
        with torch.no_grad():
            mask *= tri2d
            
        self.register_buffer("mask", mask, persistent=True)

        # 利用 Hook 机制显式截断无效区域的梯度传播
        self.weight.register_hook(lambda g: g * self.mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight * self.mask
        return F.conv2d(x, w, self.bias, stride=1, padding=self.padding, dilation=self.dilation)


class DATM_Conv(nn.Module):
    """
    顶级封装：动态自适应三角掩码卷积 (Dynamic Adaptive Triangular-Masked Convolution)
    
    物理意义：
    通过并行三个不同感受野的空洞三角分支（模拟局部、中距、全局结构），
    并通过全局池化与 MLP 结构生成动态软门控（Soft-gating）权重，
    实现基于实例内容的自适应特征聚合。
    """
    def __init__(self, in_channels: int, out_channels: int, k: int = 3):
        super(DATM_Conv, self).__init__()
        
        # 特征维度映射对齐
        self.align = nn.Conv2d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Identity()

        # 构建多尺度并行空洞三角掩码分支 (Dilation = 1, 2, 3)
        self.branch1 = Dilated_TM_Conv(out_channels, out_channels, k=k, dilation=1)
        self.branch2 = Dilated_TM_Conv(out_channels, out_channels, k=k, dilation=2)
        self.branch3 = Dilated_TM_Conv(out_channels, out_channels, k=k, dilation=3)

        # 动态路由感知机网络 (Dynamic Routing Perceptron)
        self.gap = nn.AdaptiveAvgPool2d(1)
        # 为维持极简计算量，设计了带有通道衰减的特征提纯层
        mid_channels = max(8, out_channels // 4)
        
        self.fc1 = nn.Conv2d(out_channels, mid_channels, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        # 输出维度为 3 * out_channels，对应三个空洞分支的软权重
        self.fc2 = nn.Conv2d(mid_channels, 3 * out_channels, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input feature tensor of shape [B, C_in, H, W]
        Returns:
            Tensor: Dynamically routed multi-scale tensor of shape [B, C_out, H, W]
        """
        # 1. 通道空间对齐
        x = self.align(x)
        b, c, h, w = x.size()
        
        # 2. 并行提取多尺度上三角约束特征
        feat_1 = self.branch1(x)
        feat_2 = self.branch2(x)
        feat_3 = self.branch3(x)
        
        # 3. 聚合全局多尺度上下文
        U = feat_1 + feat_2 + feat_3
        
        # 4. 生成自适应感受野的动态门控系数
        s = self.gap(U)
        z = self.relu(self.fc1(s))
        route_weights = self.fc2(z).view(b, 3, c, 1, 1) # Shape: [B, 3, C, 1, 1]
        
        # 5. 跨尺度分支 Softmax 归一化，强制特征贡献总和为 1
        route_weights = F.softmax(route_weights, dim=1)
        
        # 6. 执行数据驱动的软特征融合
        out = (feat_1 * route_weights[:, 0, :, :, :] + 
               feat_2 * route_weights[:, 1, :, :, :] + 
               feat_3 * route_weights[:, 2, :, :, :])
               
        return out


# 使用示例
if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 构建模拟医学影像特征输入张量
    input_tensor = torch.randn(1, 3, 256, 256).to(device)
    
    # 实例化 DATM-Conv 模型并载入目标设备
    model = DATM_Conv(in_channels=3, out_channels=3).to(device)
    
    print(model)
    output_tensor = model(input_tensor)
    
    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)  
    print("output_tensor_shape :", output_tensor.shape)
    
    print("\n哔哩哔哩/微信公众号: CV缝合救星, 独家整理! \n")