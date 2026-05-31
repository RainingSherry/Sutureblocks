import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class S_GDA(nn.Module):
    """
    S-GDA: Spectral-Gated Directional Attention
    频域门控方向性注意力模块
    核心思想：利用频域全局能量分布，引导空间域的非对称方向卷积与自相关特征精炼。
    """
    def __init__(self, in_channels=3, dim=64, expansion_factor=2.66, patch_size=8, bias=False):
        """
        参数:
            in_channels: 初始输入通道 (适配 3 通道图像)
            dim: 模块内部工作通道数
            expansion_factor: 门控前馈网络的通道扩张倍数
            patch_size: 频域分块自相关计算的块大小
        """
        super(S_GDA, self).__init__()
        self.dim = dim
        self.patch_size = patch_size
        hidden_dim = int(dim * expansion_factor)

        # 特征维度映射，适配外部输入
        self.entry = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1, bias=bias)

        # ---------------------------------------------------------
        # 1. 频谱门控引导分支 (Spectral Gating)
        # ---------------------------------------------------------
        self.spectral_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, 1, bias=bias),
            nn.Sigmoid()
        )

        # ---------------------------------------------------------
        # 2. 多尺度方向性非对称感知 (Directional Convolution)
        # ---------------------------------------------------------
        # 针对扫描伪影的方向性特征，采用 1x3 和 3x1 深度卷积
        self.dir_conv_h = nn.Conv2d(dim, dim, kernel_size=(1, 3), padding=(0, 1), groups=dim, bias=bias)
        self.dir_conv_v = nn.Conv2d(dim, dim, kernel_size=(3, 1), padding=(1, 0), groups=dim, bias=bias)

        # ---------------------------------------------------------
        # 3. 门控非线性特征精炼路径 (Gated Refinement)
        # ---------------------------------------------------------
        self.project_in = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, 
                                stride=1, padding=1, groups=hidden_dim * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=bias)

        # 映射回原始输出通道
        self.exit = nn.Conv2d(dim, in_channels, kernel_size=3, padding=1, bias=bias)
        
        # 学习参数：控制空间方向性特征的残差权重
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x_in):
        # 保存全局残差
        shortcut_global = x_in
        
        # 映射至内部高维空间
        x = self.entry(x_in)
        shortcut_local = x

        # ---------------------------------------------------------
        # 步骤 1: 频域频谱门控校准
        # ---------------------------------------------------------
        s_weight = self.spectral_gate(x)
        x_gated = x * s_weight

        # ---------------------------------------------------------
        # 步骤 2: 局部块自相关计算 (维纳-辛钦定理)
        # ---------------------------------------------------------
        # 分块 (Patching)
        x_patch = rearrange(
            x_gated, 'b c (h ph) (w pw) -> b c h w ph pw',
            ph=self.patch_size, pw=self.patch_size
        )
        
        # 利用 FFT 计算功率谱并执行逆变换 (IRFFT)
        Xf = torch.fft.rfft2(x_patch.float())
        power = Xf * torch.conj(Xf)
        autocorr_patch = torch.fft.irfft2(power, s=(self.patch_size, self.patch_size))
        
        # 重组回空间图 (Depatching)
        autocorr_map = rearrange(
            autocorr_patch, 'b c h w ph pw -> b c (h ph) (w pw)',
            ph=self.patch_size, pw=self.patch_size
        )

        # ---------------------------------------------------------
        # 步骤 3: 方向性增强与特征融合
        # ---------------------------------------------------------
        feat_h = self.dir_conv_h(autocorr_map)
        feat_v = self.dir_conv_v(autocorr_map)
        
        # 用门控特征加上经过方向性卷积的自相关特征
        refined_feat = x_gated + self.gamma * (feat_h + feat_v)

        # ---------------------------------------------------------
        # 步骤 4: 门控投影精炼
        # ---------------------------------------------------------
        x_proj = self.project_in(refined_feat)
        x_proj = self.dwconv(x_proj)
        
        # 通道对半切分
        x1, x2 = x_proj.chunk(2, dim=1)
        
        # GELU 激活 + 门控乘积
        out = F.gelu(x1) * x2
        out = self.project_out(out)

        # 局部残差闭环
        out = out + shortcut_local
        
        # 映射回外部输出维度并添加全局残差
        out = self.exit(out)
        return out + shortcut_global


# ==========================================
# 使用示例与测试主函数
# ==========================================
if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 模拟突发影像输入 [B, C, H, W]
    input_tensor = torch.randn(1, 3, 256, 256).to(device)
    
    # 直接实例化 S_GDA 模块，剥离任何无用的外壳
    model = S_GDA(in_channels=3).to(device)
    
    print(model)
    
    output_tensor = model(input_tensor)
    
    # 打印维度验证结果
    print("-" * 30)
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape)
    
    print("\n哔哩哔哩/微信公众号: CV缝合救星, 独家整理! \n")