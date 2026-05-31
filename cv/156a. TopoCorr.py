import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TopoCorr(nn.Module):
    r""" CV缝合救星创新模块: TopoCorr (拓扑感知几何校正模块)
    创新点: 1. 显式拓扑连通性先验 2. 动态几何度校正 3. 多极性社区交互
    """
    def __init__(self, n_communities, block_size, embed_dim, dropout=0.1):
        super(TopoCorr, self).__init__()
        self.K = n_communities    # 潜在社区数量
        self.B = block_size      # 图像补丁块大小 (例如 8x8)
        self.D = embed_dim       # 嵌入维度

        # 1. 混合成员身份映射 (Mixed-Membership Identity Mapping)
        self.pi_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(embed_dim // 2, n_communities),
            nn.Softmax(dim=-1)
        )

        # 2. 创新：动态几何度校正网络 (Dynamic Geometric Degree Correction)
        self.dyn_degree_net = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 4, kernel_size=3, padding=1, groups=embed_dim // 4),
            nn.BatchNorm2d(embed_dim // 4),
            nn.SiLU(),
            nn.Conv2d(embed_dim // 4, 1, kernel_size=1),
            nn.Sigmoid() 
        )

        # 3. 创新：多极性社区亲和矩阵 (Multi-polar Community Affinity Matrix)
        self.affinity_matrix = nn.Parameter(torch.randn(n_communities, n_communities))
        
        # 4. 创新：拓扑连通性先验生成的几何算子 (Topological Connectivity Prior Operator)
        # 修复点：确保生成 (N, N) 形状的矩阵，其中 N = B * B
        self.register_buffer('geometric_kernel', self._generate_geometric_kernel(block_size))

        self.dropout = nn.Dropout(dropout)

    def _generate_geometric_kernel(self, B):
        r""" 生成像素对之间的几何连通性矩阵 (N x N)
        """
        N = B * B
        kernel = np.zeros((N, N))
        
        # 遍历所有像素对 (i, j)，计算它们在 2D 空间中的几何距离
        for i in range(N):
            # 将一维索引转回 2D 坐标
            r_i, c_i = i // B, i % B
            for j in range(N):
                r_j, c_j = j // B, j % B
                
                # 计算像素点 i 和 j 之间的欧式距离
                dist = np.sqrt((r_i - r_j)**2 + (c_i - c_j)**2)
                # 使用高斯函数将距离转化为连通性先验
                kernel[i, j] = np.exp(-dist / (2 * (B / 4)**2))
                
        return torch.from_numpy(kernel).float()

    def forward(self, x, att_logits):
        r""" 拓扑感知几何校正的前向传播
        """
        b, h, w, d = x.shape
        n_patches = h * w
        x_flat = x.reshape(b, n_patches, d)

        # A. 提取混合成员身份 (Mixed-Membership Identities)
        pi = self.pi_net(x_flat)

        # B. 创新实现: 动态几何度校正 (Dynamic Geometric Degree Correction)
        x_img = x.permute(0, 3, 1, 2)
        dyn_theta = self.dyn_degree_net(x_img).view(b, n_patches, 1)

        # C. 创新实现: 显式拓扑连通性先验 (Explicit Topological Connectivity Prior)
        # 修复点：直接使用 (N, N) 形状进行 expand
        topo_prior = self.geometric_kernel.unsqueeze(0).expand(b, -1, -1)

        # D. 计算拓扑感知几何校正偏置 (Compute Topology-Aware Geometric Bias)
        # 1. 计算社区交互 (多极性亲和): (B, N, K) @ (K, K) @ (B, K, N) -> (B, N, N)
        comm_interaction = torch.matmul(torch.matmul(pi, self.affinity_matrix), pi.transpose(1, 2))

        # 2. 融合动态度校正与拓扑先验
        theta_ij = torch.matmul(dyn_theta, dyn_theta.transpose(1, 2))
        
        # 采用对数域计算偏置，匹配注意力 Logits 量级
        tiny_val = 1e-6
        topo_corr_bias = torch.log(theta_ij * F.relu(comm_interaction) * topo_prior + tiny_val)

        # E. 应用校正偏置
        num_heads = att_logits.shape[1]
        topo_corr_bias = topo_corr_bias.unsqueeze(1).expand(-1, num_heads, -1, -1)
        
        out_logits = att_logits + self.dropout(topo_corr_bias)
        
        return out_logits

# 使用示例
if __name__ == "__main__":
    # 强制使用 CPU 运行以避开部分环境下的 cuDNN 警告，实际使用时可切回 cuda
    device = "cpu" 
    
    B = 8             # 局部块大小 (8x8)
    K = 5             # 社区数量
    D = 64            # 维度
    N_Heads = 4       
    N_Patches = B * B # 64

    # 1. 输入特征图: (B, H, W, C)
    input_features = torch.randn(2, B, B, D).to(device)
    
    # 2. 原始自注意力 Logits: (B, Heads, N, N)
    standard_logits = torch.randn(2, N_Heads, N_Patches, N_Patches).to(device)

    # 3. 实例化 TopoCorr
    topo_corr_module = TopoCorr(n_communities=K, block_size=B, embed_dim=D).to(device)
    
    print("--- TopoCorr 模块运行验证 ---")
    corrected_logits = topo_corr_module(input_features, standard_logits)

    print("输入特征维度   :", input_features.shape)
    print("原始 Logits 维度 :", standard_logits.shape)
    print("校正后 Logits 维度:", corrected_logits.shape)
    print("\n[CV缝合救星原创]: 拓扑几何矩阵维度已匹配，模块运行正常。 \n")