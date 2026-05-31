import torch
import torch.nn as nn

"""
    基于方差统计的注意力机制：
        写作思路与代码讲解：https://www.bilibili.com/video/BV1vdGMzcET8/
        作用位置：任何单一输出特征后，或者任何即插即用模块中。
        主要功能：通过特征方差统计的空间分布特性自适应地调整特征表达，而不是直接学习权重（Sigmod函数）。
        代码使用方式与写作思路请务必看视频~  
"""

class VarianceAttentionModule(nn.Module):
    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # 计算空间维度的方差 【空间维度的方差平方】
        spatial_var = torch.var(x, dim=(-2, -1), keepdim=True).pow(2)
        # 全局方差归一化因子【全局方差统计量】
        global_var_norm = spatial_var.sum(dim=[2, 3], keepdim=True) / (H * W - 1)
        # 计算注意力系数
        attention_coef = (spatial_var / (4 * (global_var_norm + self.eps))) + 0.5
        # 生成注意力权重
        attention_weight = self.sigmoid(attention_coef)
        # 应用注意力机制
        return x * attention_weight

if __name__ == "__main__":
    x = torch.randn(1, 32, 50, 50)
    modulator = VarianceAttentionModule()
    output = modulator(x)
    print(f'输入特征尺寸: {x.size()}')
    print(f'输出特征尺寸: {output.size()}')
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：微创新·代码无误")