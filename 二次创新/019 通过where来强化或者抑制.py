import torch
import torch.nn as nn

"""
    （特征重标定）通道自适应阈值XXX注意力模块：【从即插即用模块中任意组合注意力】
        写作思路与代码讲解：https://www.bilibili.com/video/BV1NHa2zqEwv/
        作用位置：任何单一输出特征后，或者任何即插即用模块中。
	    主要功能：通道分组与自适应阈值门控，对特征图进行权重加权，增强关键特征、抑制冗余信息。
        代码层面：①将输入特征按通道维度分组；②每组特征通过独立注意力模块生成注意力图；
                ③再通过 Sigmoid 函数将注意力值归一化至 [0,1] 区间；
                ④基于样本内均值动态生成阈值，对注意力图进行门控筛选；
                ⑤拼接掩码，与原始特征逐点相乘完成重标定。
"""

# 从我主页中 找一些注意力机制插入进来即可
# 要求：输入输出不变即插即用模块
class XXXX(nn.Module):
    def __init__(self, in_channels):
        super(XXXX, self).__init__()
    def forward(self, x):
        return x

class ATA(nn.Module):
    """
    ATA: Adaptive Threshold Attention
    ------------------------------------------------
    输入:  x ∈ R^{B×C×H×W}
    过程:  (1) 按通道分组 → (2) 组内注意力(attn_block) → (3) Sigmoid
          (4) 自适应阈值门控(样本内均值) → (5) 掩码拼接 → (6) x * mask
    输出:  与输入同形状的张量, 逐点重标定后的特征
    """
    def __init__(
        self,
        channels: int,
        groups: int = 8,
    ):
        super().__init__()
        assert channels % groups == 0, \
            f"`channels`({channels}) 必须能被 `groups`({groups}) 整除"
        self.groups = groups
        self.c_per_group = channels // groups

        Plugmodel = []
        for i in range(groups):
            model = XXXX(in_channels = self.c_per_group)
            Plugmodel.append(model)

        self.attn_modules = nn.ModuleList(Plugmodel)

        # 概率化激活（注意力到 [0,1]）
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) 沿通道维分组
        groups = torch.split(x, self.c_per_group, dim=1)  # 列表长度 = self.groups
        masks = []
        for g_feat, attn in zip(groups, self.attn_modules):
            # 2) 组内注意力 (形状与 g_feat 相同)
            a = attn(g_feat)
            # 3) 概率化
            a = self.sigmoid(a)
            # 4) 自适应阈值门控: 按样本对 (C_g, H, W) 求均值得到 gate
            #    高于 gate 的位置强通过(置 1), 否则保留原值
            gate = a.mean(dim=(1, 2, 3), keepdim=True)  # [B,1,1,1]
            mk = torch.where(a > gate, torch.ones_like(a), a)
            masks.append(mk)

        # 5) 拼接各组掩码, 与输入通道对齐
        mask = torch.cat(masks, dim=1)  # [B, C, H, W]
        # 6) 重标定
        return x * mask

if __name__ == '__main__':
    x = torch.randn(1, 32, 50, 50)
    model = ATA(channels=32)
    output = model(x)
    print(f"输入张量形状: {x.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：微创新·代码无误")