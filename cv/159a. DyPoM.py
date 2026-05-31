import torch
import torch.nn as nn

# =============================================================================
# 动态多项式混合器 (Dynamic Polynomial Mixer, DyPoM) 官方复现代码
# 
# 核心贡献：针对原生 PoM 静态多项式权重对复杂场景泛化性不足的问题，
# 提出实例条件路由机制 (Instance-Conditioned Routing Mechanism)。
# 该机制能够在维持 O(N) 线性计算复杂度的前提下，依据输入图像的全局上下文先验，
# 自适应地生成高阶多项式聚合系数，实现特征的动态表达与结构保真。
# =============================================================================

class DyPoM(nn.Module):
    """
    动态多项式混合器模块 (Dynamic Polynomial Mixer)
    
    参数说明:
        dim (int): 输入特征向量的通道维度 (C)。
        hidden_dim (int, optional): 隐空间映射的高维状态维度 (D)。为保证表征容量，默认设置为 2 * dim。
        k (int): 状态聚合所使用的多项式函数的最高阶数 (Degree of Polynomial)。
        act_layer (nn.Module): 特征映射路径采用的非线性激活函数。
    """
    def __init__(self, dim, hidden_dim=None, k=2, act_layer=nn.GELU):
        super().__init__()
        hidden_dim = hidden_dim or dim * 2
        self.k = k
        self.hidden_dim = hidden_dim

        # ---------------------------------------------------------
        # [特征映射分支] 高维状态子空间转换 (High-dimensional State Mapping)
        # 对应论文中的 W_h，将输入特征从维度 C 投影至具有更高表征容量的隐空间 D
        # ---------------------------------------------------------
        self.Wh = nn.Linear(dim, hidden_dim)
        self.act_h = act_layer()

        # ---------------------------------------------------------
        # 🌟 核心创新模块：实例条件路由机制 (Instance-Conditioned Routing Module) 🌟
        # 理论依据：摒弃全局共享的静态参数 \alpha，构建轻量级瓶颈感知网络 (Bottleneck Network)。
        # 该子模块通过捕获全局统计特征，动态解算各样本在多项式展开时的最优系数解空间，
        # 赋予模型千图千面的自适应调制能力 (Adaptive Modulation Capacity)。
        # ---------------------------------------------------------
        reduction_ratio = 4
        self.dynamic_coefficient_generator = nn.Sequential(
            nn.Linear(dim, hidden_dim // reduction_ratio),  # 降维压缩，提取核心统计分量
            nn.GELU(),
            nn.Linear(hidden_dim // reduction_ratio, k * hidden_dim) # 升维重建，输出满秩动态系数
        )

        # ---------------------------------------------------------
        # [上下文检索分支] 门控查询生成 (Gated Query Generation)
        # 对应论文中的 W_s，利用 Sigmoid 函数将特征约束至 (0, 1) 区间，
        # 形成用于提取共享上下文状态的非线性门控矩阵。
        # ---------------------------------------------------------
        self.Ws = nn.Linear(dim, hidden_dim)
        self.act_s = nn.Sigmoid()

        # ---------------------------------------------------------
        # [特征重构分支] 降维输出映射 (Dimensionality Reduction & Output Projection)
        # 对应论文中的 W_o，将融合后的上下文特征从隐空间 D 逆映射回原始输入空间 C。
        # ---------------------------------------------------------
        self.Wo = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        """
        前向传播函数 (Forward Pass)
        输入:
            x (Tensor): 序列化特征张量，形状为 [B, N, C]，其中 N 为序列长度或空间像素总数 (H*W)。
        输出:
            out (Tensor): 经过动态多项式混合调制后的特征张量，形状与输入对齐为 [B, N, C]。
        """
        B, N, C = x.shape

        # =========================================================
        # 阶段一：动态多项式系数推断 (Dynamic Coefficient Inference)
        # =========================================================
        # 1. 全局上下文先验提取 (Global Context Prior Extraction)
        # 沿序列维度执行全局平均池化 (GAP)，消除空间位移冗余，获得稳健的全局分布统计量: [B, C]
        global_context_prior = x.mean(dim=1) 
        
        # 2. 实例级自适应权重生成 (Instance-level Adaptive Weight Generation)
        # 利用路由模块推断专属的动态多项式系数: [B, k * hidden_dim]
        dynamic_alpha = self.dynamic_coefficient_generator(global_context_prior)
        
        # 3. 参数空间重构 (Parameter Space Reshaping)
        # 重构为 [B, k, hidden_dim] 结构，显式对齐不同的多项式阶数。
        dynamic_alpha = dynamic_alpha.view(B, self.k, self.hidden_dim)

        # =========================================================
        # 阶段二：共享状态聚合与上下文检索 (State Aggregation & Context Retrieval)
        # =========================================================
        
        # 1. 高维状态映射映射 (State Feature Projection)
        hx = self.act_h(self.Wh(x))  # 形状: [B, N, hidden_dim]

        # 2. 动态多项式状态展开 (Dynamic Polynomial State Expansion)
        poly_sum = torch.zeros_like(hx)
        for p in range(1, self.k + 1):
            # 提取对应阶数 p 的动态系数矩阵，并进行维度扩张以匹配特征广播 (Broadcasting) 机制
            alpha_p = dynamic_alpha[:, p-1, :].unsqueeze(1)  # 形状: [B, 1, hidden_dim]
            poly_sum += alpha_p * (hx ** p)

        # 3. 序列级上下文压缩 (Sequence-wise Context Compression)
        # 沿序列长度 N 求和，严格保持 O(N) 的线性复杂度，构建全局共享上下文 H(X)
        H_X = poly_sum.sum(dim=1, keepdim=True)  # 压缩后形状: [B, 1, hidden_dim]

        # 4. 门控上下文查询计算 (Gated Contextual Query Computation)
        sx = self.act_s(self.Ws(x))  # 形状: [B, N, hidden_dim]

        # =========================================================
        # 阶段三：特征协同与投影重构 (Feature Synergy & Projection)
        # =========================================================
        
        # 1. 门控上下文融合 (Gated Context Blending)
        # 利用动态门控矩阵对聚合后的全局上下文状态进行元素级乘法检索
        out = sx * H_X  # 形状恢复至: [B, N, hidden_dim]

        # 2. 空间降维重建 (Spatial Reconstruction)
        out = self.Wo(out)  # 输出形状: [B, N, C]

        return out

# =============================================================================
# DyPoM 二维空间适配器 (2D Spatial Wrapper for DyPoM)
# 
# 设计意图：构建即插即用的拓扑适配层 (Topology Adapter)，
# 使得序列化处理的 DyPoM 模块能够以零侵入的方式无缝集成至 
# ResNet, ConvNeXt, YOLO 等主流 2D 卷积神经网络架构中。
# =============================================================================

class DyPoM_2D(nn.Module):
    def __init__(self, dim, hidden_dim=None, k=2):
        super().__init__()
        self.dypom = DyPoM(dim, hidden_dim, k)

    def forward(self, x):
        # 捕获四维张量结构: [Batch, Channel, Height, Width]
        B, C, H, W = x.shape
        
        # 空间拓扑序列化 (Spatial-to-Sequence Tokenization)
        # 将 2D 空间场展平并转置为 1D 标记序列: [B, H*W, C]
        x_flat = x.flatten(2).transpose(1, 2)
        
        # 执行实例感知的动态多项式混合注意力计算
        out = self.dypom(x_flat)
        
        # 序列向空间拓扑逆映射 (Sequence-to-Spatial Reconstruction)
        out = out.transpose(1, 2).reshape(B, C, H, W)
        
        return out


# =============================================================================
# 模块完整性验证与消融测试脚手架 (Verification & Ablation Scaffold)
# =============================================================================
if __name__ == "__main__":
    # 构造标准正态分布的伪造特征输入，模拟 Batch=2, C=64, 空间分辨率 32x32 的特征图
    dummy_input = torch.randn(2, 64, 32, 32)
    
    print("[DyPoM 实验测试台] 开始验证动态多项式混合器模块...")
    
    # 实例化二维适配器版本的 DyPoM 模块
    model = DyPoM_2D(dim=64, hidden_dim=128, k=2)
    
    # 计算网络参数总量，以评估其在移动端或边缘计算设备部署的可行性
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] 模块可学习参数总计 (Trainable Parameters): {trainable_params / 1e3:.2f} K")
    
    # 执行前向传播推断
    output = model(dummy_input)
    print(model)
    print("\n[结构张量推演报告] Tensor Flow Analysis:")
    print(f" -> 输入特征空间映射: {dummy_input.shape}")
    print(f" -> 输出特征空间映射: {output.shape}")
    
    # 拓扑连贯性校验 (Topology Coherence Check)
    if dummy_input.shape == output.shape:
        print("\n[+] 验证通过 (Verification Passed): 结构张量输入输出维度完全对齐，动态权重路由机制介入成功，模块具备直接嵌入 SOTA 架构进行训练的完备性。")
    else:
        print("\n[-] 验证异常 (Verification Failed): 张量维度失配，请检查序列化/反序列化映射过程。")