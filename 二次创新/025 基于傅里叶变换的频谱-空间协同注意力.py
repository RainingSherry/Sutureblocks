import torch
import torch.nn as nn

"""
    基于傅里叶变换的频谱-空间协同注意力：
        写作思路与代码讲解：https://www.bilibili.com/video/BV1nHd4BEEh7
        作用位置：任何单一特征处理时，或者任何即插即用模块中。
        主要功能（写作要点）：①多域信息协同增强；②目标显著性与噪声抑制；③全局–局部联合表征建模。
        代码层面：通过融合空间域与频域特征，实现“保留原有显著区域 + 频域增强细节”的特征融合，再通过注意力机制实现特征增强。
"""

class SpectralSpatialAttentionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        # 输入特征预处理：1×1卷积 + GELU
        self.input_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.GELU()
        )

        # ===== 空间通道注意力（SCA） =====
        self.sca_conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.sca_pool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化

        # ===== 两个分支卷积 =====
        self.branch_spatial = nn.Conv2d(channels, channels, kernel_size=1, stride=1, groups=1)
        self.branch_freq = nn.Conv2d(channels, channels, kernel_size=1, stride=1, groups=1)

        # 可学习参数 α 和 β
        self.alpha = nn.Parameter(torch.zeros(channels, 1, 1))  # 控制增强特征
        self.beta = nn.Parameter(torch.ones(channels, 1, 1))    # 控制原始输入保留

    def forward(self, x):
        # Step 1: 输入预处理
        feat = self.input_proj(x)

        # Step 2: 空间通道注意力 (SCA)
        sca_weights = self.sca_conv(self.sca_pool(feat))  # [B,C,1,1]
        feat_sca = sca_weights * feat                     # 加权增强

        # Step 3: 两个分支
        spatial_branch = self.branch_spatial(feat_sca)    # 分支1
        freq_branch = self.branch_freq(feat_sca)          # 频域分支2

        # Step 4: 频域调制
        freq_branch_fft = torch.fft.fft2(freq_branch, norm='backward')
        fused_freq = spatial_branch * freq_branch_fft     # 分支1 × 频域分支
        recon = torch.fft.ifft2(fused_freq, dim=(-2, -1), norm='backward')
        recon = torch.abs(recon)                          # 转回时域并取幅值

        # Step 5: 融合 (α 控制增强, β 控制原始)
        out = recon * self.alpha + feat_sca * self.beta
        return out

if __name__ == '__main__':
    input_tensor = torch.randn(1, 32, 50, 50)
    model = SpectralSpatialAttentionBlock(channels=32)
    output_tensor = model(input_tensor)
    print(f"输入张量形状: {input_tensor.shape}")
    print(f"输出张量形状: {output_tensor.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")
