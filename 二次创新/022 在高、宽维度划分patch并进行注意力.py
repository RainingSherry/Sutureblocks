import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    基于Patch分块操作的通道注意力：
        写作思路与代码讲解：https://www.bilibili.com/video/BV1Tix6z3E45/
        作用位置：任何单一特征处理时，或者任何即插即用模块中。
        主要功能（写作要点）：① Patch分块操作保留关键细节；②注意力机制区分特征差异
        代码层面：通过对输入特征图进行Patch划分（分块操作），对每个Patch空间信息进行编码，并生成通道注意力权重，最终实现对重要特征增强与无关特征抑制。
"""

class PatchwiseChannelAttention(nn.Module):
    def __init__(self, output_dim: int, patch_size: int) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.patch_size = patch_size
        # 定义第一个线性层：将每个补丁展平后的向量映射到一半维度
        self.patch_mlp_in = nn.Linear(patch_size * patch_size, output_dim // 2)
        # 对线性层输出进行归一化，稳定训练
        self.patch_norm = nn.LayerNorm(output_dim // 2)
        # 定义第二个线性层：把一半维度映射回设定的输出维度
        self.patch_mlp_out = nn.Linear(output_dim // 2, output_dim)

        # 定义1×1卷积层，用于通道融合
        self.proj_conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            前向传播流程：
                1. 输入特征 (B, C, H, W)，先调整通道维到最后 (B, H, W, C)
                2. 将图像分块为 (P×P) 补丁，得到 (B, H/P, W/P, P, P, C)
                3. 重排为 (B, H/P*W/P, P*P, C)，并在通道维上取均值
                4. 得到 (B, H/P*W/P, P*P)，再经过两层 MLP + LN，输出 (B, H/P*W/P, output_dim)
                5. 在通道维上做 softmax，得到注意力分布
                6. 将注意力作用到特征上，并重排回特征图格式
                7. 双线性插值上采样回原图大小 (B, output_dim, H, W)
                8. 最后用 1×1 卷积融合，输出结果
        """
        # 获取输入张量的形状
        B, C, H, W = x.shape
        P = self.patch_size
        # 检查 H 和 W 是否可以被 patch_size 整除
        if (H % P != 0) or (W % P != 0):
            raise ValueError(
                f"H({H}) 和 W({W}) 必须能被 patch_size P({P}) 整除；"
                "若不整除，可在前处理阶段先做 padding 或调整 P。"
            )

        # 调整通道维到最后 (B, H, W, C)
        x_hw_last = x.permute(0, 2, 3, 1)
        # 使用 unfold 以步长 P 划分补丁 -> (B, H/P, W/P, P, P, C)
        patches = x_hw_last.unfold(1, P, P).unfold(2, P, P)
        # 重排为 (B, H/P*W/P, P*P, C)
        patches = patches.reshape(B, -1, P * P, C)
        # 在通道维上取均值，得到 (B, H/P*W/P, P*P)
        patch_tokens = patches.mean(dim=-1)

        # 输入两层 MLP + LN -> (B, H/P*W/P, output_dim)
        features = self.patch_mlp_in(patch_tokens)
        features = self.patch_norm(features)
        features = self.patch_mlp_out(features)
        # 在通道维度上计算 softmax，得到注意力分布
        attn = F.softmax(features, dim=-1)
        # 应用注意力（逐元素乘法）
        features = features * attn  # (B, H/P*W/P, output_dim)

        # 恢复成特征图形式 -> (B, H/P, W/P, C_out)
        features = features.reshape(B, H // P, W // P, self.output_dim)
        # 调整为 (B, C_out, H/P, W/P)
        features = features.permute(0, 3, 1, 2)
        # 上采样回原始大小 (B, C_out, H, W)
        features = F.interpolate(features, size=(H, W), mode='bilinear', align_corners=False)
        # 最后用 1×1 卷积融合特征
        out = self.proj_conv(features)
        return out

if __name__ == '__main__':
    # 随机生成一个输入张量 (batch=2, 通道数=32, 高宽=50×50)
    input_tensor = torch.randn(2, 32, 50, 50)
    # 初始化模型，输出通道=32，补丁大小=5
    model = PatchwiseChannelAttention(output_dim=32, patch_size=5)
    output = model(input_tensor)
    print(f"输入张量形状: {input_tensor.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")