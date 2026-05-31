import torch
import torch.nn as nn
from einops import rearrange

class DGF_SMA(nn.Module):
    """
    CVPR 风格创新模块: 动态门控频率感知稀疏多头注意力 (DGF-SMA)
    创新点: 1. 动态门控融合 2. 局部频率增强投影 3. 跨模态稀疏重标定
    """
    def __init__(self, dim, num_heads, bias=False):
        super(DGF_SMA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 频率感知投影层 (Frequency-aware Projection): 结合 1x1 降维与 3x3 深度卷积提取局部高频边缘 
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        
        # 动态门控机制 (Dynamic Gating Mechanism): 学习模态间的自适应融合权重
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.Sigmoid()
        )

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        # 稀疏重标定单元 (Sparse Recalibration Unit): 增强判别性特征响应 
        self.channel_recal = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        vis_fea = x[0]  # 可见光特征 
        ir_fea = x[1]   # 红外特征 
        b, c, h, w = vis_fea.shape

        # 执行频率感知投影
        vis_qkv = self.qkv_dwconv(self.qkv(vis_fea))
        vis_q, vis_k, vis_v = vis_qkv.chunk(3, dim=1)

        ir_qkv = self.qkv_dwconv(self.qkv(ir_fea))
        ir_q, ir_k, ir_v = ir_qkv.chunk(3, dim=1)

        # 多头维度重塑
        vis_q = rearrange(vis_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        vis_k = rearrange(vis_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        vis_v = rearrange(vis_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        ir_q = rearrange(ir_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        ir_k = rearrange(ir_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        ir_v = rearrange(ir_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # 归一化以增强训练稳定性
        vis_q = torch.nn.functional.normalize(vis_q, dim=-1)
        vis_k = torch.nn.functional.normalize(vis_k, dim=-1)
        ir_q = torch.nn.functional.normalize(ir_q, dim=-1)
        ir_k = torch.nn.functional.normalize(ir_k, dim=-1)

        # 互导引交叉注意力 (Cross-modal Interaction) 
        # 使用可见光 Q 引导红外 K,V 
        attn_ir = (vis_q @ ir_k.transpose(-2, -1)) * self.temperature
        attn_ir = attn_ir.softmax(dim=-1)
        out_ir = (attn_ir @ ir_v)

        # 使用红外 Q 引导可见光 K,V 
        attn_vis = (ir_q @ vis_k.transpose(-2, -1)) * self.temperature
        attn_vis = attn_vis.softmax(dim=-1)
        out_vis = (attn_vis @ vis_v)

        # 映射回空间维度
        out_ir = rearrange(out_ir, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_vis = rearrange(out_vis, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # 动态门控融合: 自适应平衡两个模态的贡献 
        concat_fea = torch.cat([out_vis, out_ir], dim=1)
        gating_weight = self.gate(concat_fea)
        fused_fea = gating_weight * out_vis + (1 - gating_weight) * out_ir

        # 输出投影与残差重标定
        out = self.project_out(fused_fea)
        out = out * self.channel_recal(out) # 

        return out # 哔哩哔哩/微信公众号: CV缝合救星, 独家整理!

# 使用示例
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 模拟输入特征 (B, C, H, W)
    vis_in = torch.randn(1, 64, 32, 32).to(device)
    ir_in = torch.randn(1, 64, 32, 32).to(device)

    # 初始化魔改后的 CVPR 风格模块
    model = DGF_SMA(dim=64, num_heads=8, bias=False).to(device)
    
    print(f"--- DGF-SMA 模块加载成功 ---")
    output = model([vis_in, ir_in])

    # 验证维度
    print("可见光输入维度 :", vis_in.shape)
    print("红外光输入维度 :", ir_in.shape)
    print("魔改输出维度   :", output.shape)
    print("\n[创新成功]: 动态门控频率感知稀疏注意力模块已就绪。 \n")