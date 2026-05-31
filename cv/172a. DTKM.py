import torch
import torch.nn as nn
import torch.nn.functional as F

class DTKM(nn.Module):
    """
    DTKM: Dynamic Topological Knot Module
    动态拓扑结模块
    核心思想：在 CKConv 基础上引入动态感知引导与非对称编织对齐，专为红外弱小目标探测设计。
    """
    def __init__(self, in_channels=3, dim=64):
        super(DTKM, self).__init__()
        self.dim = dim
        
        # 初始特征投影：将输入通道映射到隐藏层维度
        self.entry = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1, bias=False)

        # 1. 动态拓扑感知分支：利用局部上下文生成自适应显著性权重
        self.topology_gate = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim // 4, dim, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 2. 核心中国结卷积算子（CK-Engine）：模拟水平与垂直长程编织路径
        # 分组卷积以保持轻量化特性
        self.horizontal_knot = nn.Conv2d(dim, dim, kernel_size=(1, 7), padding=(0, 3), groups=dim, bias=False)
        self.vertical_knot = nn.Conv2d(dim, dim, kernel_size=(7, 1), padding=(3, 0), groups=dim, bias=False)
        
        # 3. 编织对齐机制：同步建模多路径特征，校准子带间的分布差异
        self.weaving_alignment = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True)
        )

        # 4. 最终输出重构：将增强特征投影回目标通道空间
        self.exit = nn.Conv2d(dim, in_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x_in):
        # 步骤 1: 基础特征精炼
        x = self.entry(x_in)
        
        # 步骤 2: 生成动态拓扑感知掩码
        # 用于动态调节“中国结”采样点的显著性，抑制海面背景杂波
        t_mask = self.topology_gate(x)
        
        # 步骤 3: 非规则拓扑路径提取
        # 模拟中国结的穿插结构，分别在水平与垂直维度捕获弱小目标的分布特征
        h_feat = self.horizontal_knot(x)
        v_feat = self.vertical_knot(x)
        
        # 步骤 4: 特征编织与多尺度对齐
        # 将各路径特征聚合，并执行空间域的分布一致性校准
        weaved_feat = torch.cat([h_feat, v_feat], dim=1)
        aligned_feat = self.weaving_alignment(weaved_feat)
        
        # 步骤 5: 能量聚焦与残差耦合
        # 利用动态掩码强化目标像素，并通过残差连接保障深层梯度流动
        refined_feat = aligned_feat * t_mask + x
        
        # 输出重构
        out = self.exit(refined_feat)
        return out

# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 模拟红外舰船图像输入 [B, 3, 256, 256]
    input_tensor = torch.randn(1, 3, 256, 256).to(device)

    # 直接实例化真正的核心创新模块：DTKM
    model = DTKM(in_channels=3, dim=64).to(device)

    print(model)

    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)  

    print("output_tensor_shape :", output_tensor.shape)

    print("\n哔哩哔哩/微信公众号: CV缝合救星, 独家整理! \n")