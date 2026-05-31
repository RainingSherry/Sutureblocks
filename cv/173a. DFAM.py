import torch
import torch.nn as nn
import torch.nn.functional as F

class DFAM(nn.Module):
    """
    DFAM: Dynamic Fourier Alignment Module - 最终修复版
    动态傅里叶对齐模块
    核心思想：在频域内引入动态特征校准与多尺度幅度补偿，实现极端暗光下的高保真照度恢复。
    """
    def __init__(self, dim):
        super(DFAM, self).__init__()
        self.dim = dim
        
        # 1. 动态权重生成路径：利用空间先验预测频率修正因子
        self.dynamic_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 2. 空间增强分支：用于保留局部的精细纹理
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim, dim, 1, bias=False)
        )

        # 3. 频率对齐算子：执行特征协同对齐
        self.freq_align = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        # 输入形状 [B, C, H, W]
        B, C, H, W = x.shape
        
        # 步骤 1: 生成空间动态引导权重 [B, C, 1, 1]
        gate = self.dynamic_gate(x)
        
        # 步骤 2: 频率域处理 (Fourier Domain)
        # 执行实数快速傅里叶变换
        x_fft = torch.fft.rfft2(x, norm='ortho')
        
        # 核心修复：确保权重 gate 的形状与复数张量 x_fft 完全对齐
        # x_fft 形状为 [B, C, H, W//2 + 1]
        # 使用广播机制时，显式指定 gate 作用于通道维度
        x_fft_scaled = x_fft * gate
        
        # 逆傅里叶变换还原空间域尺寸，irfft2 返回实数 4D 张量 [B, C, H, W]
        x_freq = torch.fft.irfft2(x_fft_scaled, s=(H, W), norm='ortho')
        
        # 步骤 3: 空间细节补充与协同对齐
        x_spa = self.spatial_refine(x)
        
        # 此时 x_freq 和 x_spa 均为 4D，直接加和进行卷积
        out = self.freq_align(x_freq + x_spa)
        
        return out + x # 残差连接

class DFAM_Net(nn.Module):
    """
    顶级封装网络
    """
    def __init__(self, in_dim=3, dim=64):
        super(DFAM_Net, self).__init__()
        self.proj_in = nn.Conv2d(in_dim, dim, 3, 1, 1)
        self.dfam_module = DFAM(dim=dim)
        self.proj_out = nn.Conv2d(dim, in_dim, 3, 1, 1)

    def forward(self, x):
        x = self.proj_in(x)
        x = self.dfam_module(x)
        out = self.proj_out(x)
        return out

# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 模拟极端暗光图像输入 [1, 3, 256, 256]
    input_tensor = torch.randn(1, 3, 256, 256).to(device)

    model = DFAM_Net(in_dim=3, dim=64).to(device)

    print(model)

    output_tensor = model(input_tensor)

    # 打印维度验证
    print("\n" + "="*30)
    print("input_tensor_shape  :", input_tensor.shape)  
    print("output_tensor_shape :", output_tensor.shape)
    print("="*30)

    print("\n哔哩哔哩/微信公众号: CV缝合救星, 独家整理! \n")