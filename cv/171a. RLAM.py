import torch
import torch.nn as nn
import torch.nn.functional as F

class RLAM(nn.Module):
    """
    RLAM: Residual Lifting Alignment Module
    残差提升对齐模块
    核心思想：在提升小波变换中引入残差预测路径与多尺度子带对齐，实现隐私保护与降质恢复的深度耦合。
    """
    def __init__(self, in_channels=3, dim=64):
        super(RLAM, self).__init__()
        self.dim = dim
        
        # 初始特征映射：将 3 通道输入映射到高维特征空间
        self.entry = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1, bias=False)

        # 1. 残差预测算子：用于在提升步骤中学习局部纹理的自适应补偿
        self.predict_op = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        )
        
        # 2. 更新算子：平衡子带特征能量，确保变换在数学上的可逆性
        self.update_op = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        )

        # 3. 特征对齐层：校准隐写域中各频段子带的分布偏移
        self.alignment = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 4, kernel_size=1, groups=4, bias=False),
            nn.Sigmoid()
        )

        # 4. 最终输出投影：从特征空间映射回图像空间
        self.exit = nn.Conv2d(dim, in_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x_in):
        # 步骤 1: 基础特征提取
        x = self.entry(x_in)
        B, C, H, W = x.shape
        
        # 步骤 2: 空间采样分解（四分采样，构建提升框架基础）
        x00 = x[:, :, 0::2, 0::2] # 近似分量
        x01 = x[:, :, 0::2, 1::2] # 水平细节
        x10 = x[:, :, 1::2, 0::2] # 垂直细节
        x11 = x[:, :, 1::2, 1::2] # 对角细节

        # ----------- 提升步骤 (Lifting Steps) -----------
        # 预测：利用低频分量预测高频波动，并提取残差
        d1 = x01 - self.predict_op(x00)
        d2 = x10 - self.predict_op(x00)
        d3 = x11 - self.predict_op(x00)
        
        # 更新：利用残差信息更新低频，增强特征的抗干扰能力
        s = x00 + self.update_op(d1 + d2 + d3)

        # ----------- 特征对齐 (Alignment) -----------
        # 将分解后的子带拼接，并在隐写特征域进行动态对齐
        feat_stack = torch.cat([s, d1, d2, d3], dim=1) # [B, 4C, H/2, W/2]
        align_mask = self.alignment(feat_stack)
        feat_aligned = feat_stack * align_mask

        # ----------- 逆变换与特征精炼 (Inverse Refinement) -----------
        # 拆分对齐后的子带
        s_a, d1_a, d2_a, d3_a = torch.chunk(feat_aligned, 4, dim=1)
        
        # 执行逆提升操作，重建高质量特征
        r00 = s_a - self.update_op(d1_a + d2_a + d3_a)
        r01 = d1_a + self.predict_op(r00)
        r10 = d2_a + self.predict_op(r00)
        r11 = d3_a + self.predict_op(r00)

        # 空间重组回原始分辨率
        out_feat = torch.zeros_like(x)
        out_feat[:, :, 0::2, 0::2] = r00
        out_feat[:, :, 0::2, 1::2] = r01
        out_feat[:, :, 1::2, 0::2] = r10
        out_feat[:, :, 1::2, 1::2] = r11

        # 映射回 3 通道输出
        out = self.exit(out_feat)
        return out

# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 模拟隐私保护图像输入 [B, 3, 256, 256]
    input_tensor = torch.randn(1, 3, 256, 256).to(device)

    # 直接实例化真正的核心创新模块：RLAM
    model = RLAM(in_channels=3, dim=64).to(device)

    print(model)

    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)  

    print("output_tensor_shape :", output_tensor.shape)

    print("\n哔哩哔哩/微信公众号: CV缝合救星,独家整理! \n")