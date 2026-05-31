import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =============================================================================
# 模块名称: Spectral-Spatial Calibration Attention (S-SCA)
# 模块功能: 
# 本模块通过集成多光谱DCT变换与动态频域滤波机制，旨在解决遥感显著性检测中空间池化
# 导致的特征坍缩问题。模块利用频域能量重构实现背景与目标的有效解耦，并通过频域特征
# 生成的动态门控对空间投影分支进行协同校准，从而在抑制复杂背景的同时精确锐化目标边缘。
# =============================================================================

def get_dct_filter(tile_size, total_len, channel):
    # 生成用于DCT变换的二维基函数权重
    # 权重维度需修正为 [num_freqs * C, 1, H, W] 以适配分组卷积要求
    dct_filter = torch.zeros(total_len, tile_size, tile_size)
    c_coeff = torch.zeros(tile_size)
    c_coeff[0] = 1 * math.sqrt(1 / tile_size)
    c_coeff[1:] = 1 * math.sqrt(2 / tile_size)
    
    for i in range(tile_size):
        for j in range(tile_size):
            for x in range(tile_size):
                for y in range(tile_size):
                    dct_filter[i * tile_size + j, x, y] = \
                        c_coeff[i] * c_coeff[j] * math.cos(math.pi * i * (2 * x + 1) / (2 * tile_size)) * \
                        math.cos(math.pi * j * (2 * y + 1) / (2 * tile_size))
                        
    # 针对分组卷积配置，在输出通道维度进行重复，输入通道维度保持为1
    dct_filter = dct_filter.unsqueeze(1).repeat(channel, 1, 1, 1)
    return dct_filter

class SpectralFilter(nn.Module):
    # 动态频域滤波器：在DCT域内对频率分量进行通道级自适应调制
    def __init__(self, in_channels, tile_size=16):
        super(SpectralFilter, self).__init__()
        self.tile_size = tile_size
        self.num_freqs = tile_size * tile_size
        
        # 频率描述符映射，用于计算各频率分量的注意力权重
        self.freq_desc = nn.Sequential(
            nn.Linear(in_channels * self.num_freqs, in_channels // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, in_channels * self.num_freqs, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x_dct):
        # x_dct 维度为 [B, num_freqs, C, 1, 1]
        b, f, c, h, w = x_dct.size()
        
        # 展平特征向量以接入线性层进行全局频率建模
        context = x_dct.view(b, -1)
        freq_weights = self.freq_desc(context).view(b, f, c, 1, 1)
        
        # 应用动态权重对频率分量进行加权调制
        return x_dct * freq_weights

class S_SCA(nn.Module):
    # 频域-空间协同校准注意力模块主架构
    def __init__(self, in_channels, feat_size=16):
        super(S_SCA, self).__init__()
        self.feat_size = feat_size
        self.num_freqs = feat_size ** 2
        
        # 空间投影分支，用于保留并增强局部空间特征
        mip = max(8, in_channels // 4)
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, in_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        
        # 预计算并注册DCT基函数滤波器
        self.register_buffer('dct_filter', get_dct_filter(feat_size, self.num_freqs, in_channels))
        
        # 实例化动态频域滤波器模块
        self.spectral_filter = SpectralFilter(in_channels, feat_size)
        
        # 校准门控生成层，利用频域精炼特征生成权重图
        self.reproj_spectral = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        
        # 最终线性投影与激活层
        self.proj_final = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.act = nn.SiLU()

    def forward(self, x):
        # 获取输入张量维度
        b, c, h, w = x.size()
        
        # 空间域分支：通过降维与升维过程提取局部特征
        x_proj = self.act(self.bn1(self.conv1(x)))
        x_proj = self.act(self.bn2(self.conv2(x_proj)))
        
        # 频域协同分支：利用分组卷积在特征空间执行DCT变换
        x_dct_coeffs = F.conv2d(x, self.dct_filter, groups=c)
        # 重新排列张量形状以对齐频率与通道维度
        x_dct_coeffs = x_dct_coeffs.view(b, self.num_freqs, c, 1, 1)
        
        # 在频域内执行自适应过滤，增强目标频段信号
        x_dct_modulated = self.spectral_filter(x_dct_coeffs)
        
        # 执行逆DCT变换（IDCT）将特征映射回空间域
        x_spectral_recon = F.conv_transpose2d(x_dct_modulated.view(b, -1, 1, 1), 
                                             self.dct_filter, groups=c)
        
        # 频域-空间协同：利用频域特征生成的权重对空间特征进行像素级校准
        calibration_gate = self.reproj_spectral(x_spectral_recon)
        f_co_calibration = x_proj * calibration_gate
        
        # 采用残差连接输出最终特征，提升训练稳定性
        return self.proj_final(f_co_calibration) + x

# 验证脚本
if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 构造符合DCT变换要求的输入张量（尺寸需为2的幂次）
    input_tensor = torch.randn(1, 64, 16, 16).to(device)
    
    # 实例化S-SCA模块
    model = S_SCA(in_channels=64, feat_size=16).to(device)
    
    print(model)
    output_tensor = model(input_tensor)
    
    # 打印维度以验证算子正确性
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape)
    
    print("\n毕哩毕哩/微信公众号: CV缝合救星, 独家整理! \n")