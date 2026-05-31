import torch
import torch.nn as nn
import torch_dct as DCT
""" 
    基于离散余弦变换的高频注意力模块（DCT High Frequency Attention）：
        写作思路与代码讲解：https://www.bilibili.com/video/BV1ZgmgBGEf6/
        作用位置：任何单一特征处理时，或者任何即插即用模块中。
        主要功能（写作要点）：①频域特征的过滤；②防止大量无效的低频信息干扰深度模型的学习过程。
        代码层面：采用固定频域掩码完成低频抑制与高频选择，不依赖额外可学习参数，避免引入训练不稳定性，进而保持频率选择的一致性。
"""
class DCTHighFrequencySpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()                        # 调用父类构造函数，完成基础初始化
        self.ratio = (0.25, 0.25)                 # 设置低频区域比例，左上角 25%×25% 视为低频

    def _build_highfreq_mask(self, h, w, ratio):
        h0 = int(h * ratio[0])                    # 计算低频区域在高度方向的边界
        w0 = int(w * ratio[1])                    # 计算低频区域在宽度方向的边界
        mask = torch.ones((h, w), requires_grad=False)  # 初始化全 1 的频率掩码矩阵
        mask[:h0, :w0] = 0                        # 将左上角低频区域置 0，实现低频抑制
        return mask                               # 返回高频保留掩码

    def forward(self, x):
        _, _, H, W = x.size()                     # 读取输入特征图的空间尺寸 H 和 W
        freq_feature = DCT.dct_2d(x, norm='ortho') # 对输入特征做 2D 离散余弦变换，转换到频域
        mask = self._build_highfreq_mask(H, W, self.ratio).to(x.device) # 根据当前尺寸生成高频掩码并移动到同一设备
        mask = mask.view(1, H, W).expand_as(freq_feature) # 扩展掩码形状，使其与频域特征维度一致
        enhanced_freq = freq_feature * mask       # 在频域中抑制低频，仅保留高频信息
        highfreq_spatial_map = DCT.idct_2d(enhanced_freq, norm='ortho') # 对增强后的频域特征做逆 DCT，回到空间域
        return x * highfreq_spatial_map           # 用高频增强图作为空间权重，对原特征进行加权

if __name__ == "__main__":
    x = torch.randn(1, 64, 50, 50)
    model = DCTHighFrequencySpatialAttention()
    output = model(x)
    print(f"输入张量形状: {x.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")