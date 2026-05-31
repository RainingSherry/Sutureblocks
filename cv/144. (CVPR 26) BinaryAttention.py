import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# CV缝合救星独家复现 BinaryAttention 模块
# 参考论文 BinaryAttention: One-Bit QK-Attention for Vision and Diffusion Transformers
# 设计思路：
# 1. 将 Q 和 K 二值化 (sign)
# 2. 使用位运算代替浮点点积
# 3. 引入可学习偏置缓解信息损失
# 4. 支持端到端训练
# =========================

class BinaryAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(BinaryAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim 必须能被 num_heads 整除"
        
        # Q, K, V 映射层
        self.qkv = nn.Linear(dim, dim * 3)
        
        # 可学习偏置，用于缓解二值化信息损失
        self.bias = nn.Parameter(torch.zeros(num_heads, 1, 1))
        
        # 输出线性层
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        x: [B, N, C] -> B=batch, N=token数, C=通道
        """
        B, N, C = x.shape
        
        # 计算 Q,K,V
        qkv = self.qkv(x)  # [B, N, 3*C]
        q, k, v = qkv.chunk(3, dim=-1)  # 每个 [B, N, C]
        
        # 分头
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1,2)  # [B, heads, N, head_dim]
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1,2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1,2)
        
        # =========================
        # CV缝合救星独家二值化 Q/K
        # sign函数将 Q,K 转换为 -1/+1
        # 代替浮点点积
        # =========================
        q_bin = torch.sign(q)
        k_bin = torch.sign(k)
        
        # QK^T 位运算注意力 (乘以可学习偏置)
        attn_logits = torch.matmul(q_bin, k_bin.transpose(-2, -1)) / self.head_dim
        attn_logits = attn_logits + self.bias  # 缓解二值化损失
        
        # softmax 得到注意力权重
        attn = F.softmax(attn_logits, dim=-1)
        
        # attention加权V
        out = torch.matmul(attn, v)  # [B, heads, N, head_dim]
        
        # 合并头
        out = out.transpose(1,2).reshape(B, N, C)
        
        # 输出线性映射
        out = self.proj(out)
        
        return out

# =========================
# CV缝合救星独家测试
# =========================

if __name__ == "__main__":
    # 模拟输入
    B, N, C = 2, 16, 64  # batch 2, token 16, 通道64
    x = torch.randn(B, N, C)
    
    # 初始化 BinaryAttention
    ba = BinaryAttention(dim=C, num_heads=8)
    
    # 打印模型结构（CV缝合救星复现）
    print("=== CV缝合救星 BinaryAttention 模块结构 ===")
    print(ba)
    
    # 前向推理
    out = ba(x)
    
    # 输出形状
    print("=== CV缝合救星 BinaryAttention 输入输出信息 ===")
    print("输入 x 形状:", x.shape)
    print("输出 out 形状:", out.shape)
    
    # 随机输出前几行看看
    print("输出前两行示例:", out[0, :2, :5])