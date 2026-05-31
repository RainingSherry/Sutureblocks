import typing as t
import torch
import torch.nn as nn

""" 
    高度方向感知的多尺度空间注意力：
        写作思路与代码讲解：https://www.bilibili.com/video/BV1ihrwBNEzN/
        作用位置：任何单一特征处理时/任何普通卷积，或者任何即插即用模块中。
        主要功能（写作要点）：①方向感知的空间建模能力。②捕获多尺度空间语义信息。③保持特征在方向上一致性。（将在本视频的写作部分展开阐述）
        代码层面：沿高度/宽度方向做平均池化，将二维特征压缩为宽度/高度方向的一维特征序列；同时通过多尺度深度卷积模块构建方向特征关系，最终生成共享的 1×W 空间注意力图，对原始特征进行特定的方向性重加权。
"""
class DMSA_H(nn.Module):
    """
        DMSA-H：Height Direction-aware Multi-Scale Spatial Attention
        高度方向感知的多尺度空间注意力
    """
    def __init__(
        self,
        channels: int,                  # 输入通道数 C
        ks: t.List[int] = [3, 5, 7, 9],  # H 方向多尺度卷积核大小
        gate: str = "sigmoid",           # 门控函数：sigmoid 或 softmax
    ):
        super().__init__()

        # 通道按 4 组划分，便于多语义（多分支）建模
        assert channels % 4 == 0, "channels 必须能被 4 整除"
        assert len(ks) == 4, "ks 必须包含 4 个卷积核大小"

        self.channels = channels
        self.group_channels = channels // 4
        gc = self.group_channels

        """
            仅在卷积核大小上不同，分别负责从局部细节到长程全局趋势的多尺度空间建模，
            从而在注意力生成阶段同时保留细粒度结构与整体空间一致性。
        """
        self.dw_local = nn.Conv1d(gc, gc, kernel_size=ks[0], padding=ks[0] // 2, groups=gc)  # 局部
        self.dw_s = nn.Conv1d(gc, gc, kernel_size=ks[1], padding=ks[1] // 2, groups=gc)      # 小范围
        self.dw_m = nn.Conv1d(gc, gc, kernel_size=ks[2], padding=ks[2] // 2, groups=gc)      # 中范围
        self.dw_l = nn.Conv1d(gc, gc, kernel_size=ks[3], padding=ks[3] // 2, groups=gc)      # 大范围

        self.norm_h = nn.GroupNorm(4, channels)
        self.gate = nn.Softmax(dim=2) if gate.lower() == "softmax" else nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # -------------------------------------------------
        # 1、沿宽度做平均池化，得到高度方向的一维特征
        #  mean 掉谁，谁就消失；留下的维度，才是建模方向。
        # -------------------------------------------------
        feat_h = x.mean(dim=3)

        # -------------------------------------------------
        # 2. 按通道维将特征拆分为 4 组 每组形状：[B, C/4, W]
        # -------------------------------------------------
        g1, g2, g3, g4 = torch.split(feat_h, self.group_channels, dim=1)

        # -------------------------------------------------
        # 3. 多尺度高度方向建模 不同尺度捕捉不同感受野下的空间语义
        # -------------------------------------------------
        attn_h = torch.cat(
            (
                self.dw_local(g1), # 局部细节
                self.dw_s(g2),     # 小范围上下文
                self.dw_m(g3),     # 中范围上下文
                self.dw_l(g4)),    # 大范围上下文
            dim=1
        )

        # -------------------------------------------------
        # 4. 归一化 + 门控，生成高度方向注意力权重
        # -------------------------------------------------
        attn_h = self.gate(self.norm_h(attn_h))

        # -------------------------------------------------
        # 5. 调整形状以便与原特征相乘
        # -------------------------------------------------
        attn_h = attn_h.view(b, c, h, 1)

        # -------------------------------------------------
        # 6. 空间重加权（方向共享），仅沿高度方向变化
        # -------------------------------------------------
        return x * attn_h

if __name__ == "__main__":
    model = DMSA_H(channels=32)
    input = torch.randn(1, 32, 50, 50)
    output = model(input)
    print("input.shape:", input.shape)
    print("output.shape:", output.shape)
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")