import torch
import torch.nn as nn
from einops import rearrange

"""
    基于傅里叶变换的频域特征过滤：
        写作思路与代码讲解：https://www.bilibili.com/video/BV12dbZzwE3J/
        作用位置：任何单一输出特征后，或者任何即插即用模块中。
        主要功能：通过傅里叶变换与自学习的权重，实现频域特征过滤，保留有用频域信息。
        代码层面：1、将特征图划分为非重叠子块。
                2、对每个子块执行2D快速傅里叶变换（FFT）转换至频域。
                3、通过可学习参数对频域特征进行自适应调整。
                4、通过逆快速傅里叶变换（IFFT）转回空间域。5、最终重组回完整特征图。
"""

class Featuremodel(nn.Module):
    def __init__(self,dim):
        super(Featuremodel, self).__init__()
        self.dim = dim
    def forward(self, x):
        return x

class FreqAwareFeatureModule(nn.Module):
    """频域感知特征增强模块：通过分块傅里叶变换实现特征的频域调制"""
    def __init__(self, feat_dim, patch_size):
        super(FreqAwareFeatureModule, self).__init__()
        # 存储 patch 尺寸和特征维度
        self.patch_size = patch_size
        self.feat_dim = feat_dim

        # 可学习的频域调制参数，用于调整不同频率分量的权重
        self.freq_weight = nn.Parameter(
            torch.ones((feat_dim, 1, 1, patch_size, patch_size // 2 + 1))
        )

        self.Featue = Featuremodel(dim=feat_dim)

    def forward(self, x):
        # ①这里可以加个特征处理
        ## 说简单点：从即插即用里面找，排列组合~
        x = self.Featue(x)

        # ②或者一个跳跃连接
        res = x

        # 将输入特征图分块为非重叠子区域
        x_patched = rearrange(
            x,
            'b c (h p1) (w p2) -> b c h w p1 p2',
            p1=self.patch_size,
            p2=self.patch_size
        )

        # 对分块特征执行傅里叶变换，转换至频域
        x_freq = torch.fft.rfft2(x_patched.float())

        # 应用可学习权重调制频域特征
        x_freq_modulated = x_freq * self.freq_weight

        # 逆傅里叶变换转回空间域
        x_patched = torch.fft.irfft2(
            x_freq_modulated,
            s=(self.patch_size, self.patch_size)
        )

        # 将分块特征重组为完整特征图
        x = rearrange(
            x_patched,
            'b c h w p1 p2 -> b c (h p1) (w p2)',
            p1=self.patch_size,
            p2=self.patch_size
        )
        x = x + res
        return x

if __name__ == "__main__":
    x = torch.randn(1, 32, 64, 64)
    model = FreqAwareFeatureModule(feat_dim=32,patch_size=8)
    output = model(x)
    print(f"输入张量形状: {x.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：微创新·代码无误")