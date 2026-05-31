import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

"""
    轻量级多头交叉注意力融合模块（Lightweight Cross Attention Fusion，LCAF）：
        写作思路与代码讲解：https://www.bilibili.com/video/BV1eK2rBpEbP/
        作用位置：两个相同大小特征融合为一个特征时，或者任何即插即用模块中。
        主要功能（写作要点）：①实现主–辅特征的选择性交互；②建立跨区域的长距离依赖关系；③多模态特征的特征选择与对齐。
        代码层面：主干特征X生成Query，辅助特征Y生成Key/Value，使主分支能够在空间与通道维度上自适应地选择并对齐辅助分支中最相关信息，
                        超越传统逐元素加法或Concat的静态融合方式。
"""

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        # 可学习的权重参数，初始化为全 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        # 可学习的偏置参数，初始化为全 0
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        # 用于数值稳定性的小常数
        self.eps = eps
        # 数据格式，支持 "channels_last" 和 "channels_first"
        self.data_format = data_format
        # 检查数据格式是否合法，若不合法则抛出异常
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        # 归一化的形状
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        # 如果数据格式为 "channels_last"
        if self.data_format == "channels_last":
            # 直接调用 PyTorch 的层归一化函数
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # 如果数据格式为 "channels_first"
        elif self.data_format == "channels_first":
            # 计算通道维度上的均值
            u = x.mean(1, keepdim=True)
            # 计算通道维度上的方差
            s = (x - u).pow(2).mean(1, keepdim=True)
            # 进行归一化操作
            x = (x - u) / torch.sqrt(s + self.eps)
            # 应用可学习的权重和偏置
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

# Cross Attention Block，交叉注意力块
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        # 调用父类的构造函数
        super(CrossAttentionFusion, self).__init__()
        # 注意力头的数量
        self.num_heads = num_heads
        # 可学习的温度参数
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        # 查询卷积层
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        # 查询的深度可分离卷积层
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        # 键值卷积层
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        # 键值的深度可分离卷积层
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        # 输出投影层
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.norm = LayerNorm(dim)

    def forward(self, x, y):
        x = self.norm(x)
        y = self.norm(y)

        # 获取输入特征图的形状
        b, c, h, w = x.shape
        # 计算查询
        q = self.q_dwconv(self.q(x))

        # 计算键值
        kv = self.kv_dwconv(self.kv(y))
        # 将键值在通道维度上拆分为键和值
        k, v = kv.chunk(2, dim=1)

        # 对查询、键和值进行维度重排
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # 对查询和键进行归一化
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # 对注意力分数进行 softmax 操作
        attn = nn.functional.softmax(attn, dim=-1)
        # 计算注意力输出
        out = (attn @ v)
        # 对注意力输出进行维度重排
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        # 输出投影
        out = self.project_out(out)
        return out

if __name__ == "__main__":
    module =  CrossAttentionFusion(dim=64, num_heads=8)
    # 生成随机输入张量 x
    input_x = torch.randn(1, 64, 32, 32)
    # 生成随机输入张量 y
    input_y = torch.randn(1, 64, 32, 32)
    # 计算输出张量
    output_tensor = module(input_x,input_y)
    print('Input size:', input_x.size())
    print('Output size:', output_tensor.size())
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")