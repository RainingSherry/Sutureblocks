import torch
import torch.nn as nn

"""
    基于旋转角度位置编码的自注意力机制：
        写作思路与代码讲解：https://www.bilibili.com/video/BV1U8KxzJEgi/
        作用位置：任何Transfomer结构中。
        主要功能：通过复数旋转操作对查询 (Q) 和键 (K) 进行位置编码，能够捕捉输入特征的相对位置关系。与传统 Transformer 相比，RAPE 在保持计算效率的同时，增强了空间结构的感知能力。
        代码使用方式与写作思路请务必看视频~  
"""

def complex_rotate(input_tensor, sin_tensor, cos_tensor):
    """
    对输入张量执行复数旋转操作，用于实现旋转位置编码
    Args:
        input_tensor: 输入特征张量
        sin_tensor: 正弦值张量，用于旋转
        cos_tensor: 余弦值张量，用于旋转
    Returns:
        旋转后的张量
    """
    # 分离实部和虚部
    real_part = input_tensor[..., ::2]
    imag_part = input_tensor[..., 1::2]
    # 执行复数旋转: (a+bi)(cos+isin) = (a*cos-b*sin) + (a*sin+b*cos)i
    rotated_real = real_part * cos_tensor - imag_part * sin_tensor
    rotated_imag = real_part * sin_tensor + imag_part * cos_tensor
    # 重新组合实部和虚部
    return torch.stack([rotated_real, rotated_imag], dim=-1).flatten(-2)

class RotaryAttentionPositionEncoder(nn.Module):
    """
    基于旋转位置编码的注意力模块，增强模型对空间结构的感知能力
    """
    def __init__(self, embed_dim=256, num_heads=8, value_dim_scale=1):
        super().__init__()
        # 模型超参数
        self.value_dim_scale = value_dim_scale
        self.embed_dim = embed_dim
        self.num_attention_heads = num_heads
        self.head_dim = embed_dim * value_dim_scale // num_heads
        self.key_dim = embed_dim // num_heads
        self.scaling_factor = self.key_dim ** -0.5

        # 注意力机制所需的线性变换层
        self.query_projection = nn.Linear(embed_dim, embed_dim, bias=True)
        self.key_projection = nn.Linear(embed_dim, embed_dim, bias=True)
        self.value_projection = nn.Linear(embed_dim, embed_dim * value_dim_scale, bias=True)
        self.output_projection = nn.Linear(embed_dim * value_dim_scale, embed_dim, bias=True)

        # 初始化模型参数
        self._initialize_parameters()

        # 旋转位置编码所需的正弦和余弦参数（占位符）
        self.sin_position_encoding = nn.Parameter(torch.ones((1, 1, 1, 1)))
        self.cos_position_encoding = nn.Parameter(torch.ones((1, 1, 1, 1)))

    def forward(self, feature_map):
        batch_size, height, width, _ = feature_map.size()

        # 线性投影生成注意力机制所需的查询、键和值
        query = self.query_projection(feature_map)
        key = self.key_projection(feature_map)
        value = self.value_projection(feature_map)
        key = key * self.scaling_factor  # 缩放键值以稳定训练

        # 重塑张量以适应多头注意力机制
        query = query.view(batch_size, height, width, self.num_attention_heads, -1).permute(0, 3, 1, 2, 4)
        key = key.view(batch_size, height, width, self.num_attention_heads, -1).permute(0, 3, 1, 2, 4)

        # 应用旋转位置编码
        rotated_query = complex_rotate(query, self.sin_position_encoding, self.cos_position_encoding)
        rotated_key = complex_rotate(key, self.sin_position_encoding, self.cos_position_encoding)

        # 展平空间维度以便计算注意力
        flattened_query = rotated_query.flatten(2, 3)
        flattened_key = rotated_key.flatten(2, 3)

        # 重塑值张量并展平空间维度
        value = value.reshape(batch_size, height, width, self.num_attention_heads, -1).permute(0, 3, 1, 2, 4)
        flattened_value = value.flatten(2, 3)

        # 计算注意力分数矩阵
        attention_scores = flattened_query @ flattened_key.transpose(-1, -2)
        # 应用softmax获取注意力权重
        attention_weights = torch.softmax(attention_scores, dim=-1)
        # 计算注意力加权值
        context = torch.matmul(attention_weights, flattened_value)
        # 恢复原始空间维度布局
        context = context.transpose(1, 2).reshape(batch_size, height, width, -1)

        # 输出投影变换
        output = self.output_projection(context)
        return output

    def _initialize_parameters(self):
        """初始化模型参数"""
        # 使用Xavier初始化权重参数
        nn.init.xavier_normal_(self.query_projection.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.key_projection.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.value_projection.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.output_projection.weight)
        # 偏置项初始化为零
        nn.init.constant_(self.output_projection.bias, 0.0)

if __name__ == '__main__':
    input_tensor = torch.randn(2, 50, 50, 32)
    model = RotaryAttentionPositionEncoder(embed_dim=32)
    output_tensor = model(input_tensor)
    print(f"输入张量形状: {input_tensor.shape}")
    print(f"输出张量形状: {output_tensor.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：微创新·代码无误")