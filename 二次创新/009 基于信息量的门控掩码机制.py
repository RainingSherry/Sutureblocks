import torch
import torch.nn as nn

"""
    基于分组统计信息量门控的特征分离：
        写作思路与代码讲解：https://www.bilibili.com/video/BV1mDE5zJExT/
        作用位置：特征分解/融合位置。
        主要功能：通过信息量门控机制，实现特征通道的细粒度解耦，抑制噪声干扰，保留关键视觉信息。
        代码使用方式与写作思路请务必看视频~
"""

class GroupBatchNorm2d(nn.Module):
    """分组批量归一化层：按通道分组计算统计量"""
    def __init__(self, num_channels: int,
                 num_groups: int = 16,
                 eps: float = 1e-10):
        super().__init__()
        assert num_channels >= num_groups, "通道数必须不小于分组数"
        self.num_groups = num_groups
        self.scale = nn.Parameter(torch.randn(num_channels, 1, 1))  # 缩放参数（原gamma）
        self.shift = nn.Parameter(torch.zeros(num_channels, 1, 1))  # 偏移参数（原beta）
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape  # [批量, 通道, 高, 宽]
        # 重组为[批量, 组数, 单组元素数]以便分组统计
        grouped_feat = x.view(B, self.num_groups, -1)

        # 计算分组均值和标准差（信息量统计基础）
        group_mean = grouped_feat.mean(dim=2, keepdim=True)
        group_std = grouped_feat.std(dim=2, keepdim=True)

        # 分组归一化（抑制组内信息波动）
        normalized_feat = (grouped_feat - group_mean) / (group_std + self.eps)
        normalized_feat = normalized_feat.view(B, C, H, W)  # 恢复原形状

        return normalized_feat * self.scale + self.shift  # 缩放偏移

class InfoGatedReconstructUnit(nn.Module):
    """基于信息量的门控掩码重构单元"""
    def __init__(self,
                 in_channels: int,
                 gate_threshold: float = 0.5):
        super().__init__()
        num_groups = in_channels
        self.group_norm = GroupBatchNorm2d(in_channels, num_groups=num_groups)
        self.gate_threshold = gate_threshold  # 信息量门控阈值
        self.sigmoid = nn.Sigmoid()  # 门控激活函数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: 分组归一化（提取通道组的统计信息）
        norm_feat = self.group_norm(x)

        # Step 2: 计算通道重要性权重（基于归一化参数的分布）
        # self.group_norm.scale：是分组归一化层中可学习的缩放参数（形状为 [32, 1, 1]），每个元素对应一个通道的缩放系数。
        # self.group_norm.scale.sum()：计算所有通道缩放参数的总和，用于将权重标准化为总和为 1 的分布。
        """
            直接使用 scale 作为权重可能存在数值不稳定问题（例如不同通道的 scale 差异过大）。
            除以总和进行归一化 ===>  scale 转换为相对重要性指标
        """
        channel_weights = self.group_norm.scale / self.group_norm.scale.sum()

        # Step 3: 信息量门控计算（sigmoid激活生成连续信息量分数）
        info_scores = self.sigmoid(norm_feat * channel_weights)

        # Step 4: 生成信息量掩码（二值化筛选高/低信息量区域）
        high_info_mask = info_scores >= self.gate_threshold  # 高信息量掩码
        low_info_mask = info_scores < self.gate_threshold  # 低信息量掩码

        # Step 5: 分离特征（根据信息量掩码筛选特征）
        high_info_feat = high_info_mask * x  # 高信息量特征
        low_info_feat = low_info_mask * x  # 低信息量特征

        # total = low_info_feat + high_info_feat
        return high_info_feat,low_info_feat


if __name__ == '__main__':
    input_tensor = torch.randn(1, 32, 50, 50)  # 输入张量[批量, 通道, 高, 宽]
    model = InfoGatedReconstructUnit(in_channels=32, gate_threshold=0.5)
    HF,LF = model(input_tensor)
    print('输入尺寸:', input_tensor.shape)
    print('输出尺寸:', HF.shape)
    print('输出尺寸:', LF.shape)
    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：微创新·代码无误")