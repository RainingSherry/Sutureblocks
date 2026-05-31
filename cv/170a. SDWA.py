import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse

class SDWA(nn.Module):
    """
    SDWA: Spectral-Driven Wavelet Attention
    频域驱动的小波注意力模块
    针对突发闪烁去除任务设计的增强型注意力单元，结合了全局频谱门控与局部小波方向性。
    """
    def __init__(self, dim, num_heads=4, window_size=8, bias=False):
        super(SDWA, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        
        # 离散小波变换：使用Haar基以获取精确的方向性解耦
        self.dwt = DWTForward(J=1, wave='haar')
        self.idwt = DWTInverse(wave='haar')

        # 1. 全局频谱感知分支：通过全局池化捕获频域能量分布
        self.spectral_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 2. 局部方向性校准：针对条纹伪影处理高频子带 (LH, HL)
        self.direction_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, groups=2, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
            nn.Sigmoid()
        )

        # 3. 低频注意力投影层
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        # 缩放因子
        self.scale = (dim // num_heads) ** -0.5

    def window_partition(self, x, window_size):
        B, C, H, W = x.shape
        x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
        windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
        return windows

    def window_reverse(self, windows, window_size, H, W):
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, -1, window_size, window_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, -1, H, W)
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x

        # ----------- 步骤 1: 全局频谱门控校准 -----------
        # 捕捉全局光照振荡特征，对输入特征进行动态重加权
        spectral_weight = self.spectral_gate(x)
        x = x * spectral_weight

        # ----------- 步骤 2: 小波域多频段分解 -----------
        # 提取低频近似 LL 与高频细节 (LH, HL, HH)
        LL, Yh = self.dwt(x)
        LH, HL, HH = Yh[0][:, :, 0, :, :], Yh[0][:, :, 1, :, :], Yh[0][:, :, 2, :, :]

        # ----------- 步骤 3: 方向性高频特征提取 -----------
        # 融合水平与垂直高频分量，生成方向性权重图
        directional_mask = self.direction_conv(torch.cat([LH, HL], dim=1))

        # ----------- 步骤 4: 频域驱动的窗口注意力 -----------
        # 生成 QKV，利用方向性掩码调节 Value (V) 的响应
        qkv = self.qkv(LL)
        q, k, v = qkv.chunk(3, dim=1)
        v = v * directional_mask + v # 显式引入方向性物理先验引导

        # 窗口切分
        q_win = self.window_partition(q, self.window_size) # [nw*B, C, ws, ws]
        k_win = self.window_partition(k, self.window_size)
        v_win = self.window_partition(v, self.window_size)

        # 多头注意力计算
        nw_B, C_win, ws, _ = q_win.shape
        q_flat = q_win.view(nw_B, self.num_heads, C_win // self.num_heads, ws * ws)
        k_flat = k_win.view(nw_B, self.num_heads, C_win // self.num_heads, ws * ws)
        v_flat = v_win.view(nw_B, self.num_heads, C_win // self.num_heads, ws * ws)

        attn = (q_flat.transpose(-2, -1) @ k_flat) * self.scale
        attn = attn.softmax(dim=-1)
        out_win = (v_flat @ attn.transpose(-2, -1)).view(nw_B, C_win, ws, ws)

        # 窗口还原与投影
        LL_att = self.window_reverse(out_win, self.window_size, H // 2, W // 2)
        LL_out = self.proj_out(LL_att)

        # ----------- 步骤 5: 小波逆变换重构 -----------
        # 融合精炼后的低频特征与原始高频细节
        out = self.idwt((LL_out, [torch.stack([LH, HL, HH], dim=2)]))

        return out + shortcut

class PFESA(nn.Module):
    """
    PFESA: 适配测试主函数的顶级封装
    内部集成了 SDWA (Spectral-Driven Wavelet Attention) 模块
    """
    def __init__(self, in_channels=3, dim=32):
        super(PFESA, self).__init__()
        # 初始投影：将 3 通道映射到高维空间
        self.entry = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)
        # 核心创新模块
        self.sdwa = SDWA(dim=dim, num_heads=4, window_size=8)
        # 映射回 3 通道
        self.exit = nn.Conv2d(dim, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.entry(x)
        x = self.sdwa(x)
        x = self.exit(x)
        return x

# 使用示例
if __name__ == "__main__":
    
    # 自动选择运行设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 模拟输入：1张图片，3通道，256x256分辨率
    input_tensor = torch.randn(1, 3, 256, 256).to(device)
    
    # 实例化魔改模块
    model = PFESA().to(device)
    
    print(model)
    
    # 执行前向传播
    output_tensor = model(input_tensor)
    
    # 打印维度验证逻辑正确性
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape)
    
    print("\n哔哩哔哩/微信公众号: CV缝合救星, 独家整理! \n")