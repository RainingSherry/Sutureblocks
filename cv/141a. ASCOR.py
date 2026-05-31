import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialContextGating(nn.Module):
    """
    空间上下文门控机制 (Spatial Context Gating)
    设计动机：航空图像中的小目标极易被复杂的背景高频噪声淹没。
    该模块放置在残差连接分支，通过聚合全局的通道上下文分布来生成空间维度的注意力掩码，
    在特征相加前对恒等映射流进行自适应滤波，抑制冗余背景并突显前景目标的响应区域。
    """
    # ✅ 修复1：__init__ 拼写错误（你写成了 init）
    def __init__(self, kernel_size=7):
        super(SpatialContextGating, self).__init__()
        # 采用较大感受野的卷积核以获取更平滑的空间概率分布，避免引入过多的局部细粒度噪声
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿通道维度分别提取平均池化和最大池化特征，以保留整体上下文分布和最显著的特征响应
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # 在通道维度拼接两组空间描述符，通道数变为2
        y = torch.cat([avg_out, max_out], dim=1)
        
        # 经过卷积映射并使用Sigmoid激活，生成[0, 1]之间的空间注意力权重矩阵
        spatial_mask = self.sigmoid(self.conv(y))
        
        # 将注意力掩码广播并与输入特征逐元素相乘，实现空间级的特征重标定
        return x * spatial_mask

class ASCOR(nn.Module):
    """
    自适应空间上下文全感受野模块 (Adaptive Spatial-Context Omni-Receptive Module)

    模块介绍：
    针对无人机视角下目标尺度跨度大、分布密集的挑战，ASCOR 模块提出了一种动态多尺度特征增强范式。
    摒弃了传统的静态特征拼接(Concat)融合方式，ASCOR 引入了自适应感受野聚合机制(Dynamic Receptive Aggregation)。
    该机制包含三个异构的分支：局部细粒度感知、非对称上下文建模以及大感受野空洞探测。
    网络能够根据输入图像的内容，自适应地为不同分支分配注意力权重，实现特定尺度目标的“定制化”感受野选择。
    """
    def __init__(self, in_channels, out_channels=None, reduction=4):
        """
        初始化参数:
            in_channels: 输入特征图的通道维度
            out_channels: 输出特征图的通道维度，默认保持与输入一致以方便即插即用
            reduction: 瓶颈层的通道压缩比例，用于降低多分支并发时的计算复杂度
        """
        super(ASCOR, self).__init__()
        self.out_channels = out_channels if out_channels else in_channels
        mid_channels = in_channels // reduction
        
        # 1. 降维投影层
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True)
        )
        
        # 2. 异构特征提取分支
        # 分支 A: 局部细粒度感知分支
        self.branch_local = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True)
        )
        
        # 分支 B: 非对称上下文分支
        self.branch_asymmetric = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=(5, 1), padding=(2, 0), bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True)
        )
        
        # 分支 C: 大感受野空洞分支
        self.branch_dilated = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=5, dilation=5, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True)
        )
        
        # 3. 自适应感受野聚合模块
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        attn_channels = max(mid_channels // 2, 16) 
        
        self.attn_fc1 = nn.Conv2d(mid_channels, attn_channels, kernel_size=1, bias=False)
        self.attn_bn = nn.BatchNorm2d(attn_channels)
        # ✅ 修复2：输出通道 = mid_channels * 3（对应3个分支）
        self.attn_fc2 = nn.Conv2d(attn_channels, mid_channels * 3, kernel_size=1, bias=False)
        
        # 4. 特征重构投影层
        self.expand_conv = nn.Sequential(
            nn.Conv2d(mid_channels, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels)
        )
        
        # 5. 残差连接
        if in_channels != self.out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
        # 空间门控
        self.spatial_gating = SpatialContextGating(kernel_size=7)

    def forward(self, x):
        """前向传播逻辑"""
        identity = self.shortcut(x)
        identity = self.spatial_gating(identity)
        
        feat_reduced = self.reduce_conv(x)
        
        # 多分支提取特征
        feat_local = self.branch_local(feat_reduced)
        feat_asymm = self.branch_asymmetric(feat_reduced)
        feat_dilated = self.branch_dilated(feat_reduced)
        
        # 自适应权重生成
        fuse_sum = feat_local + feat_asymm + feat_dilated
        context_prior = self.global_pool(fuse_sum)
        
        z = F.silu(self.attn_bn(self.attn_fc1(context_prior)))
        attn_weights = self.attn_fc2(z)
        
        # ✅ 修复3：维度重塑（自动获取通道数，避免硬编码错误）
        B, C, H, W = attn_weights.shape
        # 形状变为 [B, 3, mid_channels, 1, 1]
        attn_weights = attn_weights.view(B, 3, -1, 1, 1)
        
        # 分支维度softmax
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # 加权融合
        feat_aggregated = (feat_local * attn_weights[:, 0] +
                            feat_asymm * attn_weights[:, 1] +
                            feat_dilated * attn_weights[:, 2])
        
        # 输出
        out = self.expand_conv(feat_aggregated)
        out = F.silu(out + identity)
        
        return out
    
if __name__ == "__main__":
    # 测试脚本
    batch_size = 2
    in_channels = 256
    height, width = 64, 64

    dummy_input = torch.randn(batch_size, in_channels, height, width)
    ascor_module = ASCOR(in_channels=in_channels, out_channels=128, reduction=4)

    output = ascor_module(dummy_input)
    print(ascor_module)

    print("=== ASCOR 模块性能及维度校验 ===")
    print(f"模拟输入特征张量维度: {dummy_input.shape} -> [Batch_Size, Channels, Height, Width]")
    print(f"网络输出特征张量维度: {output.shape} -> [Batch_Size, Channels, Height, Width]")

    trainable_params = sum(p.numel() for p in ascor_module.parameters() if p.requires_grad)
    print(f"ASCOR 理论可训练参数量估算: {trainable_params} Params")
    print("校验完毕，模型可即插即用部署于主流的目标检测器预测头前端。")