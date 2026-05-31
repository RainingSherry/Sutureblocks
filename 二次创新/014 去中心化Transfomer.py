import torch
import torch.nn as nn
import math

"""
    基于中心化自注意力的空间结构感知增强模块：
        写作思路与代码讲解：https://www.bilibili.com/video/BV16auBz7EqE/
        作用位置：任何单一输出特征后，或者任何即插即用模块中。
	    主要功能：通过对 Query 和 Key 的中心化与归一化，增强图像局部与全局上下文建模能力，实现一种非标准注意力机制。
        代码层面：1、将输入特征映射为序列结构（即2D→1D）。
                2、构造 Q/K/V 向量：线性变换得到 Query、Key、Value。
                3、中心化 + 归一化处理：对 Q/K 进行均值减除与平方归一化，突出结构差异。
                4、计算注意力并重构图像：通过注意力机制对 Value 加权，再还原空间结构。
"""

class SpatialPatcher(nn.Module):
    """将空间特征图转换为序列表示"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """输入: [B, C, H, W] -> 输出: [B, H*W, C]"""
        x = x.flatten(2).transpose(1, 2)  # 展平空间维度并转置
        return x

class SpatialUnpatcher(nn.Module):
    """将序列表示恢复为空间特征图"""
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
    def forward(self, x, spatial_shape):
        """输入: [B, L, C] -> 输出: [B, C, H, W]"""
        B, L, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, spatial_shape[0], spatial_shape[1])
        return x


class FeatureEnhancementModule(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.qkv_projection = nn.Linear(feature_dim, feature_dim * 3)
        self.output_projection = nn.Linear(feature_dim, feature_dim)
        self.spatial_patcher = SpatialPatcher()
        self.spatial_unpatcher = SpatialUnpatcher(embed_dim=feature_dim)
        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, x):
        # 保存原始空间尺寸
        spatial_shape = (x.shape[2], x.shape[3])

        # 转换为序列表示: [B, C, H, W] -> [B, H*W, C]
        x_seq = self.spatial_patcher(x)

        # 生成QKV
        B, seq_len, C = x_seq.shape
        qkv = self.qkv_projection(x_seq)
        query, key, value = torch.split(qkv, C, dim=2)

        # 特征中心化 （消除区域均值偏差）
        query = query - query.mean(dim=2, keepdim=True)
        key = key - key.mean(dim=2, keepdim=True)

        # 特征平方与归一化
        query_sq = torch.pow(query, 2)
        key_sq = torch.pow(key, 2)

        # 特征维度归一化 (沿通道、沿空间归一化)
        query_sq = query_sq / (query_sq.sum(dim=2, keepdim=True) + 1e-7)
        key_sq = key_sq / (key_sq.sum(dim=2, keepdim=True) + 1e-7)
        query_sq = nn.functional.normalize(query_sq, dim=-1)
        key_sq = nn.functional.normalize(key_sq, dim=-2)

        # 注意力计算 (无softmax)
        enhanced_features = query_sq @ (key_sq.transpose(-2, -1) @ value) / math.sqrt(seq_len)

        # 恢复空间结构: [B, H*W, C] -> [B, C, H, W]
        output = self.spatial_unpatcher(enhanced_features, spatial_shape)
        return output

if __name__ == "__main__":
    x = torch.randn(1, 32, 50, 50)
    model = FeatureEnhancementModule(feature_dim=32)
    output = model(x)
    print(f'输入特征尺寸: {x.size()}')
    print(f'输出特征尺寸: {output.size()}')
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：微创新·代码无误")