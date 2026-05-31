import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Module: Triangular-Masked Convolution (TM-Conv)
# 三角掩码卷积核心算子
# 
# Abstract:
# 针对真实世界 sRGB 图像去密写（Demosaicing）过程中产生的空间相关噪声问题，
# 本模块打破了常规卷积核的全邻域感知约束。通过在卷积权重上施加严格的上三角
# 二进制掩码（Upper-triangular Mask），并在反向传播中截断相应区域的梯度，
# 从底层算子级别确保了感受野的单向（对角线方向）渐进式扩展。
# 这是构建符合真实噪声几何特性的“钻石型（Diamond-shaped）”盲点网络的核心。
# =============================================================================

class TM_Conv(nn.Conv2d):
    """
    三角掩码卷积 (Triangular-Masked Convolution)
    
    设计动机：
    标准二维卷积会聚合中心像素周围所有方向的特征，这使得自监督去噪网络容易陷入
    “恒等映射（Identity Mapping）”的陷阱。TM-Conv 强制剥离卷积核的下半部分响应，
    保证网络只能感知目标像素上方的上下文信息，从而有效隔绝空间相关的噪声信号。
    """
    def __init__(self, in_channels: int, out_channels: int, k: int):
        # 1. 继承标准二维卷积，维持输入输出的空间分辨率不变 (padding = k//2)
        super().__init__(in_channels, out_channels, kernel_size=k,
                         stride=1, padding=k//2, dilation=1, groups=1, bias=True)

        # 2. 构造与卷积核权重张量 (Weight Tensor) 同维度的全 1 掩码
        mask = torch.ones_like(self.weight)
        
        # 3. 生成基于核大小 k 的二维上三角矩阵 (Upper-triangular Matrix)
        # torch.triu 保留主对角线及以上的元素，其余强制置为 0
        tri2d = torch.triu(torch.ones(k, k, dtype=mask.dtype, device=mask.device))
        
        # 4. 利用广播机制 (Broadcasting) 将上三角约束应用到所有输入输出通道
        with torch.no_grad():
            mask *= tri2d
            
        # 5. 将掩码注册为持久化缓冲区 (Persistent Buffer)
        # 物理意义：它不是可学习参数，但会随模型状态字典 (state_dict) 保存，并同步设备转移
        self.register_buffer("mask", mask, persistent=True)

        # 6. 核心机制：注册梯度反向传播钩子 (Backward Hook)
        # 确保在优化器更新时，被掩盖区域（即下三角区域）的梯度被严格置为 0，
        # 彻底阻断非掩码区域的参数学习，维持严密的物理约束。                                                                                                                           # 哔哩哔哩/微信公众号: CV缝合救星, 独家整理!
        self.weight.register_hook(lambda g: g * self.mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape [B, C_in, H, W]
        Returns:
            Tensor: Masked-convoluted tensor of shape [B, C_out, H, W]
        """
        # 前向推演：在执行滑动窗口内积计算前，对当前权重施加物理掩码约束
        w = self.weight * self.mask
        
        # 使用 F.conv2d 执行底层张量运算
        return F.conv2d(x, w, self.bias, stride=1, padding=self.padding)


# =============================================================================
# Validation Script
# =============================================================================
if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 构造模拟输入特征张量: Batch=1, Channels=64, H=128, W=128
    input_tensor = torch.randn(1, 64, 128, 128).to(device)
    
    # 实例化 TM-Conv 算子 (输入64通道，输出128通道，卷积核大小为3)
    model = TM_Conv(in_channels=64, out_channels=128, k=3).to(device)
    
    print("[INFO] Model Architecture:")
    print(model)
    
    # 检查掩码矩阵的正确性
    print("\n[INFO] TM-Conv Mask Matrix Snapshot (k=3):")
    print(model.mask[0, 0].cpu().numpy()) 
    # 预期输出一个上三角矩阵：
    # [[1. 1. 1.]
    #  [0. 1. 1.]
    #  [0. 0. 1.]]
    
    # 执行前向推演计算
    output_tensor = model(input_tensor)
    
    # 计算算子的可学习参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 打印维度验证
    print("\n--- 维度检查 (CV缝合保底认证) ---")
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape) 
    print(f"Trainable Parameters: {total_params}")
    
    print("哔哩哔哩/微信公众号: CV缝合救星, 独家整理! \n")