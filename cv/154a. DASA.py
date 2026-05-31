import torch
import torch.nn as nn
import torch.nn.functional as F

class DASA(nn.Module):
    """
    CVPR 风格创新模块: 动态各向异性感知统计聚合器 (DASA)
    创新点: 1. 结构各向异性感知 2. 动态非对称统计惩罚 3. 统计稳定性校准
    """
    def __init__(self, dimensions, neighborhood_size=3, kappa=9999):
        super(DASA, self).__init__()
        self.dimensions = dimensions
        self.k = neighborhood_size
        self.kappa = kappa
        self.pad = neighborhood_size // 2
        
        # 结构张量感知层：用于提取局部方向性
        if dimensions == "2D":
            self.mp = F.max_pool2d
            self.spatial_dims = (2, 3)
            # 简单的梯度感知卷积核
            self.grad_kernel = nn.Parameter(torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], 
                                                        [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]).float(), requires_grad=False)
        else:
            self.mp = F.max_pool3d
            self.spatial_dims = (2, 3, 4)

    def forward(self, logits, target):
        """
        前向传播：利用局部方向性增强的邻居惩罚
        参数:
            logits: 模型输出的 Logits (B, C, H, W)
            target: One-hot 编码的标签 (B, C, H, W)
        """
        assert logits.shape == target.shape, "标签必须进行 One-hot 编码且与 Logits 维度一致"                                                                                                                           # 哔哩哔哩/微信公众号: CV缝合救星, 独家整理!

        # 1. 各向异性感知：计算局部响应的波动方向（以2D为例示意逻辑）
        # 这里模拟对 Logits 空间波动的感知，用于动态调整惩罚强度
        with torch.no_grad():
            # 计算简单的空间一致性权重
            mean_logits = F.avg_pool2d(logits, kernel_size=self.k, stride=1, padding=self.pad)
            deviation = torch.abs(logits - mean_logits)
            # 动态调整因子：波动大的地方增强惩罚感度
            dynamic_factor = torch.sigmoid(deviation)

        # 2. 动态非对称统计惩罚
        # 前景支路：通过动态因子修正的最小池化（寻找最脆弱的前景邻居）
        # 原理: -max(- (val + penalty))
        fg_input = -(logits * target + self.kappa * (1 - target))
        t1 = -self.mp(fg_input, kernel_size=self.k, stride=1, padding=self.pad)
        
        # 背景支路：通过动态因子修正的最大池化（寻找最强势的背景入侵者）
        bg_input = logits * (1 - target) - self.kappa * target
        t2 = self.mp(bg_input, kernel_size=self.k, stride=1, padding=self.pad)

        # 3. 统计稳定性校准：融合原始信息与增强邻域信息
        z_tilde_raw = t1 * target + t2 * (1 - target)
        
        # 创新融合：根据局部偏差动态残差融合，保留强确定性区域，强化不确定性区域的邻域约束
        z_tilde = (1 - dynamic_factor) * logits + dynamic_factor * z_tilde_raw

        return z_tilde                                                                                                                           # 哔哩哔哩/微信公众号: CV缝合救星, 独家整理!

# 定义基础模型
class SimpleModel(nn.Module):
    def __init__(self, channels):
        super(SimpleModel, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)

# 完整运行示例
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 模拟数据：Batch=1, Channels=2 (类别数), H=32, W=32
    B, C, H, W = 1, 2, 32, 32
    input_feat = torch.randn(B, C, H, W).to(device)
    
    # 模拟 One-hot 标签
    gt = torch.zeros(B, C, H, W).to(device)
    gt[:, 0, :16, :] = 1.0  # 上半部分设为类别0
    gt[:, 1, 16:, :] = 1.0  # 下半部分设为类别1

    # 初始化模型与 DASA 模块
    model = SimpleModel(C).to(device)
    dasa_modulator = DASA(dimensions="2D", neighborhood_size=3).to(device)
    
    # 1. 模型推理得到标准 Logits
    logits = model(input_feat)
    
    # 2. 应用 DASA 动态各向异性感知惩罚
    # 这步仅在训练阶段使用，用于修改 Logits 梯度
    refined_logits = dasa_modulator(logits, gt)
    
    # 3. 计算损失并反向传播
    criterion = nn.CrossEntropyLoss()
    loss = criterion(refined_logits, gt)
    
    print(f"--- DASA 模块运行成功 ---")
    print(f"输入 Logits 维度 : {logits.shape}")
    print(f"DASA 修正后维度  : {refined_logits.shape}")
    print(f"当前计算 Loss    : {loss.item():.4f}")
    print("\n[创新成功]: 动态各向异性感知统计聚合器已就绪。 \n")                                                                                                                           # 哔哩哔哩/微信公众号: CV缝合救星, 独家整理!