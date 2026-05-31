import torch.nn as nn
import torch

"""
    基于单头交叉注意力的多模态/单模态双特征融合：
        写作思路与代码讲解：https://www.bilibili.com/video/BV14J44zBEhh/
        作用位置：任何两个相同大小的特征融合时，或者任何即插即用模块中。
        主要功能（写作要点）：①跨层次语义与细节互补；②全局与局部信息协同；
        代码层面：交叉注意力形式的一种变种/魔改。【不要局限于某一模块的单一作用】
"""

class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim):
        super(CrossAttentionFusion, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q_feature, K_feature):
        # 输入形状: [B, C, H, W]
        B, C, H, W = Q_feature.shape
        # ========== [1] 调整为 [B, HW, C] ==========
        Q_flat = Q_feature.permute(0, 2, 3, 1).reshape(B, -1, C)  # [B, HW, C]
        K_flat = K_feature.permute(0, 2, 3, 1).reshape(B, -1, C)  # [B, HW, C]

        # ========== [2] 线性变换 ==========
        Q = self.query(Q_flat)  # [B, HW, C]
        K = self.key(K_flat)    # [B, HW, C]
        V = self.value(K_flat)  # [B, HW, C]

        # ========== [3] 注意力计算 ==========
        d_k = C ** 0.5
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / d_k  # [B, HW, HW]
        attention_weights = self.softmax(attention_scores)
        attended_features = torch.matmul(attention_weights, V)  # [B, HW, C]

        # ========== [4] 还原回 [B, C, H, W] ==========
        out = attended_features.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out

if __name__ == "__main__":
    module =  CrossAttentionFusion(embed_dim=32)
    input_x = torch.randn(2, 32, 50, 50)
    input_y = torch.randn(2, 32, 50, 50)
    output_tensor = module(input_x,input_y)
    print('Input size:', input_x.size())
    print('Output size:', output_tensor.size())
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")
