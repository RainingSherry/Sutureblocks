import torch
import torch.nn as nn
import torch.fft as fft

class DCFSA(nn.Module):
    """
    DCFSA: Dynamic Cross-Frequency Synergistic Attention (动态跨频协同注意力)
    """
    def __init__(self, in_channels, reduction=16):
        """
        参数:
            in_channels (int): 输入特征的通道数
            reduction (int): 动态频率路由器的降维比例，控制参数量
        """
        super(DCFSA, self).__init__()
        self.activation = nn.Sigmoid()  # 归一化注意力权重
        self.eps = 1e-5
        # 极轻量级 MLP，根据输入的全局统计信息，动态输出每个通道的最佳频率截断比例
        # 哔哩哔哩/微信公众号: CV缝合救星, 独家整理!
        mid_channels = max(1, in_channels // reduction)
        self.dynamic_router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()  # 将预测因子映射到 (0, 1) 区间
        )
        
        # 基础截止频率，设为可学习参数以赋予网络自适应微调的能力
        self.base_scale = nn.Parameter(torch.tensor(0.1))

    def _edge_attention(self, x):
        # 物理意义：保留原生模块的无参特性，通过局部方差最大化增强对梯度敏感的解剖轮廓
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)  # batch, c, h, w                                                                                                                           # 哔哩哔哩/微信公众号: CV缝合救星, 独家整理!
        x_var = x.var(dim=[2, 3], keepdim=True)  # batch, c, 1, 1
        y = x_minus_mu_square / (x_var + self.eps)  # batch, c, h, w
        return y

    def _structure_attention(self, x):
        # 物理意义：利用低频特征的能量重分配，抑制低频伪影干扰并强化主体形态
        energy_low = torch.pow(x, 2)  # batch, c, h, w
        energy_mu = torch.mean(energy_low, dim=[2, 3], keepdim=True)  # batch, c, 1, 1
        energy_var = torch.var(energy_low, dim=[2, 3], keepdim=True)  # batch, c, 1, 1                                                                                                                           # 哔哩哔哩/微信公众号: CV缝合救星, 独家整理!
        y = (energy_low - energy_mu) / (energy_var + self.eps)  # batch, c, h, w
        y = self.activation(y)
        return y

    def _create_dynamic_low_freq_mask(self, x, h, w):
        """ 创建通道自适应的动态低频掩码 """
        b, c, _, _ = x.size()
        device = x.device
        
        # 1. 动态生成每个通道专属的高斯截止比例 [B, C, 1, 1]
        dynamic_factor = self.dynamic_router(x)
        # 结合基础比例因子，并设置 0.05 的下界防止频率截断崩溃
        mask_ratio = (self.base_scale * dynamic_factor + 0.05) * (min(h, w) / max(h, w))
        
        # 2. 生成中心对称的空间坐标网格 [1, 1, H, W]
        # 哔哩哔哩/微信公众号: CV缝合救星, 独家整理!
        y_coord = torch.linspace(-1, 1, h, device=device)
        x_coord = torch.linspace(-1, 1, w, device=device)
        Y, X = torch.meshgrid(y_coord, x_coord, indexing='ij')
        grid_sq = (Y ** 2 + X ** 2).view(1, 1, h, w)
        
        # 3. 广播机制生成动态掩码 [B, C, H, W]
        # 核心亮点：由于 mask_ratio 不同，C 维度的每一个通道都会获得一个形状完全不同的高斯掩膜！
        mask = torch.exp(-grid_sq / (2 * mask_ratio ** 2 + self.eps))
        return mask

    def forward(self, x):
        b, c, h, w = x.size()

        # **1. 傅里叶变换进入频域**
        x_freq = fft.fftn(x, dim=(-2, -1))  # 只对 H, W 维度进行 FFT
        x_freq = fft.fftshift(x_freq, dim=(-2, -1)) # 将频域零点移动到中心

        # **2. 生成动态信道感知频域掩码**
        # 哔哩哔哩/微信公众号: CV缝合救星, 独家整理!
        low_freq_mask = self._create_dynamic_low_freq_mask(x, h, w) 
        high_freq_mask = 1.0 - low_freq_mask  # 互补的高频掩码

        low_freq = x_freq * low_freq_mask
        high_freq = x_freq * high_freq_mask

        # **3. 逆傅里叶变换回到空域**
        low_freq = torch.abs(fft.ifftn(fft.ifftshift(low_freq, dim=(-2, -1)), dim=(-2, -1)))  # batch, c, h, w
        high_freq = torch.abs(fft.ifftn(fft.ifftshift(high_freq, dim=(-2, -1)), dim=(-2, -1)))  # batch, c, h, w                                                                                                                           

        # **4. 分别提取双路统计无参注意力**
        low_structure_att = self._structure_attention(low_freq)
        high_edge_att = self._edge_attention(high_freq)

        # 🌟 魔改创新二：跨频协同互增强 🌟
        # 原版直接相加: out_att = low_edge_att + high_edge_att
        # 魔改版: 利用低频结构约束高频边缘(过滤孤立高频噪点)，利用高频边缘锐化低频结构(增强连通性)
        synergy_att = high_edge_att * (1 + low_structure_att) + low_structure_att * (1 + high_edge_att)

        # 激活并重新调制原始输入
        out_att = self.activation(synergy_att)

        return out_att * x


# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(1, 3, 256, 256).to(device)
    model = DCFSA(in_channels=3).to(device)

    print(model)
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape) 
    print("\哔哩哔哩/微信公众号: CV缝合救星, 独家整理! \n")