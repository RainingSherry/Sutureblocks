import torch
import torch.nn as nn

""" 
    基于空洞卷积的多尺度上下文增强模块：
        写作思路与代码讲解：https://www.bilibili.com/video/BV1J9BnBHEsH/
        作用位置：任何单一特征处理时/任何普通卷积，或者任何即插即用模块中。
        主要功能（写作要点）：①扩大感受野而不降低分辨率。②弥补传统卷积局限，在保持边缘定位精度并提高对大尺度结构与整体布局的感知能力。
        代码层面：1×1 卷积用于进行局部通道重组，作为基础特征；
                三个采用不同系数的 3×3 的空洞卷积分支，在不显著增加参数量的前提下引入不同尺度的上下文信息。
                在通道维进行拼接后，通过 3×3 卷积进行统一融合，压缩多尺度上下文并映射回原始通道数，得到最终输出特征。

"""

#  Multi-Scale Context Enhancement Module
#  多尺度上下文增强模块
class MSCEM(nn.Module):                       # 定义一个基于多尺度膨胀卷积的特征提取模块
    def __init__(self, in_c):                        # 初始化函数，in_c 表示输入特征的通道数
        super().__init__()                           # 调用父类 nn.Module 的初始化方法

        self.c1 = nn.Sequential(                    # 定义第一条分支：1×1 标准卷积分支
            nn.Conv2d(in_c, in_c, kernel_size=1, padding=0, bias=False),  # 使用 1×1 卷积调整通道信息，不改变空间尺寸
            nn.BatchNorm2d(in_c),                   # 对卷积输出进行批归一化，稳定训练
            nn.ReLU(inplace=True)                   # 使用 ReLU 激活函数增强非线性表达能力
        )

        self.c2 = nn.Sequential(                    # 定义第二条分支：膨胀率为 6 的 3×3 膨胀卷积分支
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=6, dilation=6, bias=False),  # 扩大感受野以捕获中尺度上下文信息
            nn.BatchNorm2d(in_c),                   # 对膨胀卷积结果进行归一化处理
            nn.ReLU(inplace=True)                   # 引入非线性激活
        )

        self.c3 = nn.Sequential(                    # 定义第三条分支：膨胀率为 12 的 3×3 膨胀卷积分支
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=12, dilation=12, bias=False), # 进一步扩大感受野，建模更大范围结构信息
            nn.BatchNorm2d(in_c),                   # 批归一化以稳定特征分布
            nn.ReLU(inplace=True)                   # 非线性激活增强表达能力
        )

        self.c4 = nn.Sequential(                    # 定义第四条分支：膨胀率为 18 的 3×3 膨胀卷积分支
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=18, dilation=18, bias=False), # 获取大尺度上下文信息，增强全局感知能力
            nn.BatchNorm2d(in_c),                   # 对特征进行归一化
            nn.ReLU(inplace=True)                   # 使用 ReLU 激活函数
        )

        self.c5 = nn.Conv2d(                        # 定义特征融合卷积层
            in_c * 4,                               # 输入通道数为四个分支特征拼接后的通道数
            in_c,                                   # 输出通道数恢复为原始通道数
            kernel_size=3,                          # 使用 3×3 卷积进行特征融合
            padding=1,                              # 填充保证输出空间尺寸不变
            dilation=1,                             # 普通卷积，不使用膨胀
            bias=False                              # 不使用偏置项
        )

    def forward(self, x):
        x1 = self.c1(x)   # (1, 32, 50, 50) 通过1×1卷积分支提取局部通道特征
        x2 = self.c2(x)   # (1, 32, 50, 50) 通过膨胀率为 6  的分支提取中尺度上下文特征
        x3 = self.c3(x)   # (1, 32, 50, 50) 通过膨胀率为 12 的分支提取更大尺度上下文特征
        x4 = self.c4(x)   # (1, 32, 50, 50) 通过膨胀率为 18 的分支提取全局感受野特征
        xc = torch.cat([x1, x2, x3, x4], axis=1)    # 在通道维度上拼接四个尺度的特征
        x = self.c5(xc)
        return x

if __name__ == '__main__':
    input = torch.rand(1, 32, 50, 50)
    model = MSCEM(32)
    output = model(input)
    print("input.shape:", input.shape)
    print("output.shape:", output.shape)
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")