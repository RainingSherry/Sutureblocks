import torch.nn as nn
import torch
import torch.nn.functional as F

"""
    基于余弦相似度进行特征融合：
        写作思路与代码讲解：https://www.bilibili.com/video/BV1vVGkzoED7/
        作用位置：同一个物体的两种特征形式。
        主要功能：首先计算特征图 A 和特征图 B在通道维度的余弦相似度，将该相似度结果作为权重作用于特征图 B，
                    再将加权后的特征图 B 与特征图 A 相加，再经过 ReLU 函数，从而得到融合后的特征结果 。
"""

class FeatureAdjuster(nn.Module):
    def __init__(self):
        super(FeatureAdjuster, self).__init__()
        self.relu_activation = nn.ReLU()

    def forward(self, feature_a, feature_b):
        # 获取 feature_a 和 feature_b 的形状信息
        shape_a, shape_b = feature_a.size(), feature_b.size()
        # 断言 feature_a 和 feature_b 在通道数必须相同
        assert shape_a[1] == shape_b[1]
        # 计算 feature_a 和 feature_b 在通道特征上的余弦相似度
        cosine_similarity = F.cosine_similarity(feature_a, feature_b, dim=1)
        # 在余弦相似度结果上增加一个维度，方便后续的乘法运算
        cosine_similarity = cosine_similarity.unsqueeze(1)
        # 将 feature_b 乘以余弦相似度结果，然后加到 feature_a 上，更新 feature_a
        feature_a = feature_a + feature_b * cosine_similarity
        # 对更新后的 feature_a 应用 ReLU 激活函数
        feature_a = self.relu_activation(feature_a)
        # 返回更新并经过激活的 feature_a
        return feature_a

if __name__ == '__main__':
    block = FeatureAdjuster()
    fa = torch.rand(1, 32, 50, 50)
    fb = torch.rand(1, 32, 50, 50)
    adjusted_feature_a = block(fa, fb)
    print(fa.size())
    print(fb.size())
    print(adjusted_feature_a.size())
    print("抖音、B站、小红书、CSDN同号")
    print("布尔大学士 提醒您：微创新·代码无误")