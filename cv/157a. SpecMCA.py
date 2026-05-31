import torch
import torch.nn as nn
from einops import rearrange

class SpecMCA(nn.Module):
    r""" CVPR 2026 风格魔改模块: SpecMCA (光谱感知多尺度色彩注意力)
    改进点: 1. 动态自适应温度调节 2. 局部空间几何补偿 3. 光谱-空间联合门控
    """
    def __init__(self, dim, num_heads, bias=False):
        super(SpecMCA, self).__init__()
        self.num_heads = num_heads
        
        # 创新1: 动态温度预测 (由输入内容决定注意力平滑度)
        self.dynamic_temp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, num_heads * 2, kernel_size=1),
            nn.Sigmoid()
        )

        self.q_proj = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=2, groups=dim, bias=bias)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=2, bias=bias)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        # 创新2: 局部几何补偿分支 (解决全局变换导致的边缘退化)
        self.local_refinement = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias),
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        )

        self.a_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=2, groups=dim, bias=bias),
            nn.Conv2d(dim, dim//2, kernel_size=1)
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        
        # A. 动态计算温度系数
        temps = self.dynamic_temp(x) # (B, H*2, 1, 1)
        temp_a, temp_v = torch.chunk(temps, 2, dim=1)

        # B. 多尺度投影
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x) * x # 融合原始输入保持高频细节
        a = self.a_proj(x)

        # C. 维度重排 (通道注意力视角)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        a = rearrange(a, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # 归一化以保证训练稳定
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        a = torch.nn.functional.normalize(a, dim=-1)

        # D. 光谱-通道注意力核心逻辑 (基于动态温度)
        attn_a = (q @ a.transpose(-2, -1)) * temp_a.view(b, self.num_heads, 1, 1)
        attn_a = attn_a.softmax(dim=-1)

        attn_k = (a @ k.transpose(-2, -1)) * temp_v.view(b, self.num_heads, 1, 1)
        attn_k = attn_k.softmax(dim=-1)
        
        # 特征聚合
        out_v = (attn_k @ v)
        out = (attn_a @ out_v)

        # E. 空间还原与局部几何补偿
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        
        # 融合局部细化分支 (创新点: 防止全局转换引起的颜色光晕)
        local_feat = self.local_refinement(x)
        out = out + local_feat 

        out = self.project_out(out)
        return out

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 输入: (B, C, H, W)
    input_tensor = torch.randn(1, 64, 128, 128).to(device)
    model = SpecMCA(dim=64, num_heads=8).to(device)
    
    print("--- SpecMCA 模块运行验证 ---")
    output = model(input_tensor)
    print("输入维度 :", input_tensor.shape)   
    print("输出维度 :", output.shape) 
    print("\n[CV缝合救星原创]: 光谱感知多尺度色彩注意力模块已就绪。\n")