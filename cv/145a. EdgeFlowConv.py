import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeFlowConv(nn.Module):
    """
    EdgeFlowConv 模块（方向流边缘增强卷积）
    魔改 DEGConv，CVPR 风格创新设计：
    1. 多尺度方向卷积（水平/垂直/斜向）
    2. 可学习方向权重门控
    3. 残差增强输出
    4. 轻量化 1x1 压缩
    """

    def __init__(self, in_channels, out_channels, reduction=4):
        super(EdgeFlowConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # ----------------- STEP 0: 通道压缩 -----------------
        self.compress = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.act = nn.ReLU(inplace=True)

        # ----------------- STEP 1: 多方向卷积 -----------------
        # 水平卷积
        self.conv_h = nn.Conv2d(in_channels // reduction, in_channels // reduction,
                                kernel_size=(1, 3), padding=(0, 1), bias=False)
        # 垂直卷积
        self.conv_v = nn.Conv2d(in_channels // reduction, in_channels // reduction,
                                kernel_size=(3, 1), padding=(1, 0), bias=False)
        # 斜向卷积（+45°）
        self.conv_d1 = nn.Conv2d(in_channels // reduction, in_channels // reduction,
                                 kernel_size=3, padding=1, bias=False)
        # 斜向卷积（-45°）
        self.conv_d2 = nn.Conv2d(in_channels // reduction, in_channels // reduction,
                                 kernel_size=3, padding=1, bias=False)

        # ----------------- STEP 2: 可学习方向权重门控 -----------------
        # 4 个方向的门控权重
        self.gate = nn.Conv2d(in_channels // reduction, 4, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # ----------------- STEP 3: 拼接 + 输出卷积 -----------------
        self.expand = nn.Conv2d((in_channels // reduction) * 4, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        """
        x: 输入特征 (B, C, H, W)
        """

        # ------ STEP 0: 通道压缩 ------
        x_shrink = self.act(self.compress(x))  # 减少计算量，提高效率

        # ------ STEP 1: 多方向卷积 ------
        feat_h = self.act(self.conv_h(x_shrink))    # 水平特征
        feat_v = self.act(self.conv_v(x_shrink))    # 垂直特征
        feat_d1 = self.act(self.conv_d1(x_shrink))  # 斜 +45°
        feat_d2 = self.act(self.conv_d2(x_shrink))  # 斜 -45°

        # ------ STEP 2: 门控融合 ------
        gate_weights = self.sigmoid(self.gate(x_shrink))  # 输出 shape=(B,4,H,W)
        feat_h = feat_h * gate_weights[:, 0:1, :, :]      # 水平加权
        feat_v = feat_v * gate_weights[:, 1:2, :, :]      # 垂直加权
        feat_d1 = feat_d1 * gate_weights[:, 2:3, :, :]    # 斜 +45°
        feat_d2 = feat_d2 * gate_weights[:, 3:4, :, :]    # 斜 -45°

        # ------ STEP 3: 拼接 + 卷积输出 ------
        combined = torch.cat([feat_h, feat_v, feat_d1, feat_d2], dim=1)
        out = self.expand(combined)

        # ------ STEP 4: 残差增强 ------
        if self.in_channels == self.out_channels:
            out = out + x  # 残差连接

        return out


# --------------------- 测试 EdgeFlowConv 模块 ---------------------
if __name__ == "__main__":
    dummy_input = torch.randn(2, 64, 256, 256)  # batch=2, 64通道, 256x256
    edgeflow = EdgeFlowConv(in_channels=64, out_channels=64)

    print("=== CVPR 风格魔改 | EdgeFlowConv 模块结构 ===")
    print(edgeflow)

    output = edgeflow(dummy_input)
    print("\n=== 输入形状 ===", dummy_input.shape)
    print("=== 输出形状 ===", output.shape)