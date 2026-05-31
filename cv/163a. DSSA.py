import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =============================================================================
# 🧠 DSSA: Dynamic Sparse-Spatial Adaptive Module (动态稀疏空间自适应模块)
# 💡 创新点：
# 1. 并行异构算子池：整合标准卷积、空洞卷积、可变形卷积与深度卷积捕获多尺度偏置。
# 2. 动态路由门控（Dynamic Gating）：自适应分配各卷积分支权重，实现实例级特征选择。
# 3. 坐标协同注意力（Coordinate Attention）：注入空间位置先验，增强跨通道信息流。
# =============================================================================

class DeformConv2d(nn.Module):
    """
    可变形卷积算子 (Deformable Convolution v2)
    通过学习偏移量(Offset)使卷积核形状自适应目标形变。
    """
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=True):
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        # 偏移量生成分支
        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        p = self._get_p(offset, dtype)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + g_rb.unsqueeze(dim=1) * x_q_rb + g_lb.unsqueeze(dim=1) * x_q_lb + g_rt.unsqueeze(dim=1) * x_q_rt

        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1).unsqueeze(dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)
        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1), torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1), indexing='ij')
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0).view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(torch.arange(1, h * self.stride + 1, self.stride), torch.arange(1, w * self.stride + 1, self.stride), indexing='ij')
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        p_n = self._get_p_n(N, dtype)
        p_0 = self._get_p_0(h, w, N, dtype)
        return p_0 + p_n + offset

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w, c = x.size(3), x.size(1)
        x = x.contiguous().view(b, c, -1)
        index = q[..., :N] * padded_w + q[..., N:]
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        return x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)], dim=-1)
        return x_offset.contiguous().view(b, c, h * ks, w * ks)

class CoordAttention(nn.Module):
    """
    坐标协同注意力 (Coordinate Attention)
    通过双向池化显式建模水平与垂直方向的空间上下文。
    """
    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()
        self.conv_h = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        return identity * a_w * a_h

class DSSA(nn.Module):
    def __init__(self, in_channels):
        super(DSSA, self).__init__()
        # 并行算子分支
        self.k1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels), nn.SiLU())
        self.k3 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1), nn.BatchNorm2d(in_channels), nn.SiLU())
        self.dk3 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=3, dilation=3), nn.BatchNorm2d(in_channels), nn.SiLU())
        self.dfk3 = DeformConv2d(in_channels, in_channels, kernel_size=3, padding=1, modulation=True)
        self.bn_df = nn.BatchNorm2d(in_channels)
        self.dwk3 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels), nn.BatchNorm2d(in_channels), nn.SiLU())
        
        # 创新点1：动态路由门控 (Dynamic Gating)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.router = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, 4), # 对应 4 个主要特征分支
            nn.Softmax(dim=1)
        )

        # 创新点2：坐标注意力增强
        self.ca = CoordAttention(in_channels, in_channels)
        
        # 最后的特征对齐与融合
        self.proj = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        # 预处理
        x_base = self.k1(x)
        
        # 并行异构算子提取
        f1 = self.k3(x_base)
        f2 = self.dk3(x_base)
        f3 = F.silu(self.bn_df(self.dfk3(x_base)))
        f4 = self.dwk3(x_base)
        
        # 计算动态分支权重 [B, 4, 1, 1]
        b, c, _, _ = x.size()
        context = self.global_pool(x_base).view(b, c)
        gate_weights = self.router(context).view(b, 4, 1, 1, 1)
        
        # 分支自适应聚合
        feats = torch.stack([f1, f2, f3, f4], dim=1) # [B, 4, C, H, W]
        f_dynamic = torch.sum(feats * gate_weights, dim=1)
        
        # 坐标感知空间增强
        f_ca = self.ca(f_dynamic)
        
        # 残差投影融合
        out = self.proj(f_ca) + x
        return out

# 使用示例
if __name__ == "__main__":
    # 配置运行环境
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 构造模拟输入 (Batch=1, Channels=3, Size=256x256)
    input_tensor = torch.randn(1, 3, 256, 256).to(device)
    
    # 初始化 DSSA 模块 (针对输入 3 通道)
    # 注意：在 CVPR 风格的主函数中，我们将模块实例化为 DSSA
    model = DSSA(in_channels=3).to(device)
    
    print("--- 动态稀疏空间自适应模块 DSSA (CVPR 2026 魔改版) ---")
    print(model)
    
    # 前向推演
    output_tensor = model(input_tensor)
    
    # 打印维度验证
    print("\ninput_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape) 
    
    print("\n哔哩哔哩/微信公众号: CV缝合救星, 独家整理! \n")