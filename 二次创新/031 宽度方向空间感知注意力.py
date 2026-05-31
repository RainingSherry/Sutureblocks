import typing as t
import torch
import torch.nn as nn
""" 
    宽度方向感知的多尺度空间注意力：
        写作思路与代码讲解：https://www.bilibili.com/video/BV1ihrwBNEzN/
        作用位置：任何单一特征处理时/任何普通卷积，或者任何即插即用模块中。
        主要功能（写作要点）：①方向感知的空间建模能力。②捕获多尺度空间语义信息。③保持特征在方向上一致性。（将在本视频的写作部分展开阐述）
        代码层面：沿高度/宽度方向做平均池化，将二维特征压缩为宽度/高度方向的一维特征序列；同时通过多尺度深度卷积模块构建方向特征关系，最终生成共享的 1×W 空间注意力图，对原始特征进行特定的方向性重加权。
"""
class DMSA_W(nn.Module):
    """
        DMSA-W：Width Direction-aware Multi-Scale Spatial Attention
        宽度方向感知的多尺度空间注意力
    """
    def __init__(
        self,
        channels: int,                     # 输入特征的通道数 C
        ks: t.List[int] = [3, 5, 7, 9],     # 多尺度 1D 卷积核大小（宽度方向）
        gate: str = "sigmoid",              # 注意力门控方式：sigmoid 或 softmax
    ):
        super().__init__()

        # 要求通道数必须能被 4 整除，便于分组建模
        assert channels % 4 == 0, "channels 必须能被 4 整除"
        assert len(ks) == 4, "ks 必须包含 4 个卷积核大小"

        self.channels = channels
        self.group_channels = channels // 4
        gc = self.group_channels

        """
            仅在卷积核大小上不同，分别负责从局部细节到长程全局趋势的多尺度空间建模，
            从而在注意力生成阶段同时保留细粒度结构与整体空间一致性。
        """
        # 局部尺度分支：捕捉细粒度、局部空间变化
        self.dw_local = nn.Conv1d(
            gc, gc, kernel_size=ks[0], padding=ks[0] // 2, groups=gc
        )
        # 小范围上下文分支
        self.dw_ms = nn.Conv1d(
            gc, gc, kernel_size=ks[1], padding=ks[1] // 2, groups=gc
        )
        # 中范围上下文分支
        self.dw_mm = nn.Conv1d(
            gc, gc, kernel_size=ks[2], padding=ks[2] // 2, groups=gc
        )
        # 大范围上下文分支：建模长程空间依赖
        self.dw_ml = nn.Conv1d(
            gc, gc, kernel_size=ks[3], padding=ks[3] // 2, groups=gc
        )
        self.norm_w = nn.GroupNorm(4, channels)
        self.gate = nn.Softmax(dim=2) if gate.lower() == "softmax" else nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 解析输入特征的维度
        b, c, h, w = x.shape
        # -------------------------------------------------
        # 1、沿高度做平均池化，得到宽度方向的一维特征描述子
        #  mean 掉谁，谁就消失；留下的维度，才是建模方向。
        # -------------------------------------------------
        feat_w = x.mean(dim=2)

        # -------------------------------------------------
        # 2. 按通道维将特征拆分为 4 组 每组形状：[B, C/4, W]
        # -------------------------------------------------
        f1, f2, f3, f4 = torch.split(feat_w, self.group_channels, dim=1)

        # -------------------------------------------------
        # 3. 多尺度宽度方向建模 不同尺度捕捉不同感受野下的空间语义
        # -------------------------------------------------
        attn_w = torch.cat(
            (
                self.dw_local(f1),  # 局部细节
                self.dw_ms(f2),     # 小范围上下文
                self.dw_mm(f3),     # 中范围上下文
                self.dw_ml(f4),     # 大范围上下文
            ),
            dim=1,
        )

        # -------------------------------------------------
        # 4. 归一化 + 门控，生成宽度方向注意力权重
        # -------------------------------------------------
        attn_w = self.gate(self.norm_w(attn_w))

        # -------------------------------------------------
        # 5. 调整形状以便与原特征相乘
        # -------------------------------------------------
        attn_w = attn_w.view(b, c, 1, w)

        # -------------------------------------------------
        # 6. 空间重加权（方向共享），仅沿宽度方向变化
        # -------------------------------------------------
        return x * attn_w

if __name__ == "__main__":
    model = DMSA_W(channels=32)
    input = torch.randn(1, 32, 50, 50)
    output = model(input)
    print("input.shape:", input.shape)
    print("output.shape:", output.shape)
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")
