import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F

class ComplexLinear(nn.Linear):
    r""" 专为复数输入设计的线性层：处理频域特征
    """
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super(ComplexLinear, self).__init__(in_features, out_features, False, device, dtype)                                                                                                                           # 哔哩哔哩/微信公众号: CV缝合救星, 独家整理!

    def forward(self, x):
        # 将复数视为实数对进行变换
        x = torch.view_as_real(x).transpose(-2, -1)
        x = torch.nn.functional.linear(x, self.weight).transpose(-2, -1)
        x = torch.view_as_complex(x.contiguous())
        return x
    
class SpecMorph(nn.Module):
    r""" CVPR 风格创新模块: SpecMorph (频域-形态学动态调制器)
    创新点: 1. 频域结构张量计算 2. 动态形态学空间校准 3. 相位自适应增强
    """
    def __init__(self, dim, neighborhood_size=3, proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.qkv_spec = ComplexLinear(dim, dim * 3)
        
        # 创新：动态形态学门控分支 (用于空间域校准)
        self.morph_gate = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, kernel_size=1)
        )
        
        # 创新：相位增强因子
        self.phase_weight = nn.Parameter(torch.ones(1, 1, 1, dim))
        
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, n, c = x.shape
        h = w = int(n ** 0.5)

        # 1. 空间域特征准备与形态学感知
        identity = x
        x_img = x.reshape(b, h, w, c).permute(0, 3, 1, 2) # (B, C, H, W)
        m_gate = self.morph_gate(x_img).permute(0, 2, 3, 1).reshape(b, n, c) # (B, N, C)

        # 2. 频域转换与结构张量计算
        # 计算 2D 实际傅里叶变换
        x_spec = torch.fft.rfft2(x.reshape(b, h, w, c), dim=(1, 2), norm='ortho')
        qkv_spec = self.qkv_spec(x_spec)
        q_s, k_s, v_s = torch.chunk(qkv_spec, chunks=3, dim=-1)

        # 创新：利用共轭乘法提取结构化统计特性，并引入相位校准
        # 对应论文中的 BCCB 投影，但增加了相位权重调整
        attn_spec = torch.conj(q_s) * k_s * self.phase_weight.to(q_s.dtype)
        
        # 3. 双域融合调制
        # 转换回空间域：Equation 15 & 16 的变体增强
        attn_spatial = torch.fft.irfft2(attn_spec, s=(h, w), dim=(1, 2), norm='ortho')
        attn_spatial = attn_spatial.reshape(b, n, c).softmax(dim=1)
        
        # 频域重投影聚合
        attn_reproj = torch.fft.rfft2(attn_spatial.reshape(b, h, w, c), dim=(1, 2))
        out_spec = torch.conj(attn_reproj) * v_s
        out_spatial = torch.fft.irfft2(out_spec, s=(h, w), dim=(1, 2), norm='ortho').reshape(b, n, c)

        # 4. 最终调制输出：结合空间形态学门控与频域聚合特征
        out = (out_spatial * torch.sigmoid(m_gate)) + identity
        out = self.proj(out)
        
        return self.drop(out)                                                                                                                           # 哔哩哔哩/微信公众号: CV缝合救星, 独家整理!

# 使用示例
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 输入维度: (Batch, Tokens, Channels)
    input_tensor = torch.randn(1, 1024, 64).to(device)
    
    # 实例化 SpecMorph 模块
    model = SpecMorph(dim=64).to(device)
    
    print("--- SpecMorph 模块运行验证 ---")
    output_tensor = model(input_tensor)
    print(model)

    print("输入维度 :", input_tensor.shape)   
    print("输出维度 :", output_tensor.shape) 
    print("\n[创新成功]: 频域-形态学动态调制器已就绪。 \n")                                                                                                                           # 哔哩哔哩/微信公众号: CV缝合救星, 独家整理!