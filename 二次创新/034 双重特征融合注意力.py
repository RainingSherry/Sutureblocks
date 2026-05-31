import torch
import torch.nn as nn

""" 
   双重特征融合注意力：
        写作思路与代码讲解：https://www.bilibili.com/video/BV1A5QCBWELo/
        作用位置：任何单一特征处理时/任何普通卷积，或者任何即插即用模块中。
        主要功能（写作要点）：①全局-局部特征的平衡与互补建模。②细粒度边缘与纹理突变信息的增强。（将在本视频的写作部分展开阐述）
        代码层面：通过空间域、傅里叶域分支并行捕获特征，并结合注意力自适应融合，实现了全局结构与局部细节的协同增强。
"""

class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups

        self.freq_channel_conv = nn.Conv2d(
            in_channels=in_channels * 2,
            out_channels=out_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=self.groups,
            bias=False
        )
        self.freq_batch_norm = nn.BatchNorm2d(out_channels * 2)
        self.freq_activation = nn.ReLU(inplace=True)

    def forward(self, input_feature):
        batch_size, num_channels, feature_height, feature_width = input_feature.size()

        # 1. 输入特征做二维实数傅里叶变换
        frequency_feature = torch.fft.rfft2(input_feature, norm='ortho')

        # 2. 提取实部和虚部
        frequency_real_part = torch.unsqueeze(torch.real(frequency_feature), dim=-1)
        frequency_imag_part = torch.unsqueeze(torch.imag(frequency_feature), dim=-1)

        # 3. 将实部和虚部拼接
        frequency_feature = torch.cat((frequency_real_part, frequency_imag_part), dim=-1)

        # 4. 调整维度，便于后续卷积处理
        frequency_feature = frequency_feature.permute(0, 1, 4, 2, 3).contiguous()
        frequency_feature = frequency_feature.view((batch_size, -1) + frequency_feature.size()[3:])

        # 5. 在频域上进行通道卷积、归一化和激活
        frequency_feature = self.freq_channel_conv(frequency_feature)
        frequency_feature = self.freq_activation(self.freq_batch_norm(frequency_feature))

        # 6. 恢复为复数张量格式
        frequency_feature = frequency_feature.view((batch_size, -1, 2) + frequency_feature.size()[2:])
        frequency_feature = frequency_feature.permute(0, 1, 3, 4, 2).contiguous()
        frequency_feature = torch.view_as_complex(frequency_feature)

        # 7. 逆傅里叶变换，恢复到空间域
        reconstructed_feature = torch.fft.irfft2(
            frequency_feature,
            s=(feature_height, feature_width),
            norm='ortho'
        )
        return reconstructed_feature

class FFA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FFA, self).__init__()

        # 空间域分支
        self.spatial_branch_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )

        # 傅里叶域分支
        self.fourier_branch = FourierUnit(in_channels, out_channels)

        # 融合注意力生成器
        self.fusion_attention_generator = nn.Sequential(
            nn.Conv2d(2 * out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, input_feature):
        # 1. 空间域特征
        spatial_branch_feature = self.spatial_branch_conv(input_feature)

        # 2. 傅里叶域特征
        fourier_branch_feature = self.fourier_branch(input_feature)

        # 3. 多分支特征拼接
        fused_multi_branch_feature = torch.cat(
            [
                spatial_branch_feature,
                fourier_branch_feature
            ],
            dim=1
        )

        # 4. 生成融合注意力图
        fusion_attention_map = self.fusion_attention_generator(fused_multi_branch_feature)

        # 5. 对各分支进行加权融合
        weighted_fusion_feature = 0
        for branch_feature in [
            spatial_branch_feature,
            fourier_branch_feature
        ]:
            weighted_fusion_feature = weighted_fusion_feature + branch_feature * fusion_attention_map

        return weighted_fusion_feature

if __name__ == "__main__":
    x = torch.randn(1, 32, 50, 50)
    model = FFA(in_channels=32, out_channels=32)
    output = model(x)
    print(f"输入张量形状: {x.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：微创新·代码无误")