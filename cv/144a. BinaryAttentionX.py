import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# CVPR 风格创新版 BinaryAttentionX 模块
# 创新点：
# 1. 可学习温度缩放增强注意力可控性
# 2. 全局上下文增强分支 (Context-Enhanced Branch)
# 3. 残差增强输出
# 4. 保留 1-bit Q/K 位运算加速
# =========================
class BinaryAttentionX(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(BinaryAttentionX, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim 必须能被 num_heads 整除"
        
        # QKV 线性映射
        self.qkv = nn.Linear(dim, dim * 3)
        
        # 可学习偏置
        self.bias = nn.Parameter(torch.zeros(num_heads, 1, 1))
        
        # 可学习温度参数
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        
        # 输出线性映射
        self.proj = nn.Linear(dim, dim)
        
        # 上下文增强分支线性层
        # 修复：输入是 head_dim 而不是 dim
        self.context_fc = nn.Linear(self.head_dim, self.head_dim)

    def forward(self, x):
        """
        x: [B, N, C] -> B=batch, N=token数, C=通道数
        """
        B, N, C = x.shape
        
        # =========================
        # 1. QKV 线性映射
        # =========================
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # =========================
        # 2. 分头
        # =========================
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1,2)  # [B, heads, N, head_dim]
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1,2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1,2)
        
        # =========================
        # 3. 1-bit 二值化 Q/K
        # =========================
        q_bin = torch.sign(q)
        k_bin = torch.sign(k)
        
        # =========================
        # 4. 注意力计算 + 可学习偏置
        # =========================
        attn_logits = torch.matmul(q_bin, k_bin.transpose(-2,-1)) / self.head_dim
        attn_logits = attn_logits + self.bias
        
        # =========================
        # 5. 温度缩放 softmax
        # =========================
        attn = F.softmax(attn_logits / self.temperature, dim=-1)
        
        # =========================
        # 6. Attention 加权 V
        # =========================
        out = torch.matmul(attn, v)
        
        # =========================
        # 7. 上下文增强分支
        #    对 V 做全局平均池化 -> 线性 -> 广播融合回输出
        # 修复输入维度匹配问题
        # =========================
        context = v.mean(dim=2, keepdim=True)                  # [B, heads, 1, head_dim]
        context = self.context_fc(context)                    # [B, heads, 1, head_dim]
        out = out + context                                    # 融合全局上下文增强信息
        
        # =========================
        # 8. 合并多头
        # =========================
        out = out.transpose(1,2).reshape(B, N, C)
        
        # =========================
        # 9. 输出线性 + 残差增强
        # =========================
        out = self.proj(out) + x
        
        return out

# =========================
# CVPR 魔改测试
# =========================
if __name__ == "__main__":
    B, N, C = 2, 16, 64
    x = torch.randn(B, N, C)
    
    # 初始化 BinaryAttentionX
    ba_x = BinaryAttentionX(dim=C, num_heads=8)
    
    # 打印模型结构
    print("=== CVPR 魔改 BinaryAttentionX 模块结构 ===")
    print(ba_x)
    
    # 前向推理
    out = ba_x(x)
    
    # 打印输入输出信息
    print("输入 x 形状:", x.shape)
    print("输出 out 形状:", out.shape)
    print("输出前两行示例:", out[0, :2, :5])