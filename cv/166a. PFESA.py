import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

# =============================================================================
# 🧠 PFESA: Progressive Frequency-Edge Selective Attention
# 💡 创新点：
# 1. 跨域交互引导：利用频域全局信息生成动态门控，自适应调节空间边缘的响应强度。
# 2. 渐进式选择融合：通过选择性注意力机制，在像素级动态加权空域细节与频域结构。
# 3. 语义增强感知：引入轻量级非线性特征映射，提升复杂航空背景下小目标的判别力。
# =============================================================================

def autopad(k, p=None, d=1):
    # 根据卷积核大小自动计算填充，保持输出形状一致
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    # 标准卷积单元：卷积 + 批归一化 + SiLU激活
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SelectiveFusion(nn.Module):
    # 选择性融合单元：通过注意力机制自适应合并不同域的特征
    def __init__(self, dim):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim * 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, f_spatial, f_freq):
        # 聚合特征并生成权重
        combined = f_spatial + f_freq
        weights = self.fc(self.avg_pool(combined))
        w1, w2 = torch.split(weights, [f_spatial.size(1), f_freq.size(1)], dim=1)
        # 像素级加权融合
        return f_spatial * w1 + f_freq * w2

class ScharrEdgeExtractor(nn.Module):
    # 空间边缘提取器：使用Scharr算子捕获高频梯度细节
    def __init__(self, channel):
        super(ScharrEdgeExtractor, self).__init__()
        
        # 定义Scharr卷积核参数
        kx = np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=np.float32)
        ky = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=np.float32)
        
        # 注册卷积核为不可训练权重
        kx = torch.from_numpy(kx).view(1, 1, 3, 3).repeat(channel, 1, 1, 1)
        ky = torch.from_numpy(ky).view(1, 1, 3, 3).repeat(channel, 1, 1, 1)
        
        self.conv_x = nn.Conv2d(channel, channel, 3, padding=1, groups=channel, bias=False)
        self.conv_y = nn.Conv2d(channel, channel, 3, padding=1, groups=channel, bias=False)
        
        self.conv_x.weight.data = kx
        self.conv_y.weight.data = ky
        self.conv_x.requires_grad = False
        self.conv_y.requires_grad = False

    def forward(self, x):
        grad_x = self.conv_x(x)
        grad_y = self.conv_y(x)
        # 简化的能量图计算
        return torch.abs(grad_x) + torch.abs(grad_y)

class PFESA(nn.Module):
    # PFESA主模块：渐进式频域-边缘选择性注意力
    def __init__(self, in_channels=3):
        super(PFESA, self).__init__()
        
        # 通道压缩与映射，平衡计算复杂度
        mid_ch = 32 if in_channels == 3 else in_channels
        self.pre_conv = Conv(in_channels, mid_ch, 3)

        # 空间路径
        self.spatial_extractor = ScharrEdgeExtractor(mid_ch)
        self.spatial_refine = nn.Sequential(
            Conv(mid_ch, mid_ch, 3),
            Conv(mid_ch, mid_ch, 3, d=2, p=2) # 空洞卷积增大感受野
        )

        # 频率路径
        self.freq_refine = nn.Sequential(
            Conv(mid_ch * 2, mid_ch * 2, 3),
            Conv(mid_ch * 2, mid_ch * 2, 1)
        )
        self.freq_post = Conv(mid_ch, mid_ch, 3)

        # 跨域交互与融合模块
        self.cross_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_ch, mid_ch, 1),
            nn.Sigmoid()
        )
        self.selective_fusion = SelectiveFusion(mid_ch)
        
        self.final_proj = Conv(mid_ch, in_channels, 1)

    def forward(self, x):
        b, c, h, w = x.size()
        
        # 初始特征映射
        feat = self.pre_conv(x)

        # --- 频率分支操作 ---
        # 1. 执行傅里叶变换
        f_freq = torch.fft.rfft2(feat, norm='ortho')
        f_real = torch.real(f_freq).unsqueeze(-1)
        f_imag = torch.imag(f_freq).unsqueeze(-1)
        
        # 2. 频域特征重组与卷积增强
        f_freq_cat = torch.cat((f_real, f_imag), dim=-1)
        f_freq_cat = rearrange(f_freq_cat, 'b c h w d -> b (c d) h w').contiguous()
        f_freq_cat = self.freq_refine(f_freq_cat)
        
        # 3. 逆变换回空域
        f_freq_back = rearrange(f_freq_cat, 'b (c d) h w -> b c h w d', d=2).contiguous()
        f_freq_back = torch.view_as_complex(f_freq_back)
        f_freq_spatial = torch.fft.irfft2(f_freq_back, s=(h, w), norm='ortho')
        f_freq_final = self.freq_post(f_freq_spatial)

        # --- 空间分支操作 (受频域引导) ---
        # 利用频域特征生成的全局门控来增强空域显著区域
        gate = self.cross_gate(f_freq_final)
        f_edge = self.spatial_extractor(feat)
        f_edge = self.spatial_refine(f_edge * gate + feat)

        # --- 渐进式选择融合 ---
        f_out = self.selective_fusion(f_edge, f_freq_final)
        
        return self.final_proj(f_out)

# 使用示例
if __name__ == "__main__":
    # 配置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 模拟航空图像输入 [Batch, Channel, Height, Width]
    input_tensor = torch.randn(1, 3, 256, 256).to(device)
    
    # 实例化PFESA模块
    model = PFESA(in_channels=3).to(device)

    print(model)
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)  
    print("output_tensor_shape :", output_tensor.shape)
    
    print("\n哔哩哔哩/微信公众号: CV缝合救星, 独家整理! \n")