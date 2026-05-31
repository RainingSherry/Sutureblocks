import torch
import torch.nn as nn
import torch.nn.functional as F

class WindowStd_Enhanced(nn.Module):
    """
    改进型窗口标准差统计模块：支持多尺度统计感知
    """
    def __init__(self, kernel_size=3, channels=None, eps=1e-5):
        super(WindowStd_Enhanced, self).__init__()
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.channels = channels
        self.eps = eps
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        
        if self.channels is not None:
            self._init_weight()

    def _init_weight(self):
        """初始化均值卷积核"""
        kernel_h, kernel_w = self.kernel_size
        kernel_area = kernel_h * kernel_w
        single_kernel = torch.ones(1, 1, kernel_h, kernel_w) / kernel_area
        self.register_buffer('mean_kernel', single_kernel.repeat(self.channels, 1, 1, 1))

    def forward(self, x):
        channels = x.shape[1]
        if self.channels is None:
            self.channels = channels
            self._init_weight()
        
        # 镜像padding保证边缘一致性
        x_padded = F.pad(x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0]), mode='reflect')
        
        # 计算窗口内均值 E[x]
        mean = F.conv2d(x_padded, weight=self.mean_kernel, groups=channels)
        
        # 计算窗口内平方的均值 E[x²]
        x_squared_padded = F.pad(x**2, (self.padding[1], self.padding[1], self.padding[0], self.padding[0]), mode='reflect')
        mean_squared = F.conv2d(x_squared_padded, weight=self.mean_kernel, groups=channels)
        
        # 标准差计算：sqrt(E[x²] - (E[x])² + eps)
        std = torch.sqrt(torch.clamp(mean_squared - mean**2, min=self.eps))
        return std

class AS_GWM(nn.Module):
    """
    CVPR 风格创新模块: 自适应统计-几何加权调制器 (AS-GWM)
    创新点: 1. 多尺度统计导引 2. 局部对比度几何约束 3. 全局-局部权重校准
    """
    def __init__(self, ch):
        super(AS_GWM, self).__init__()
        
        # 1. 局部结构提取分支
        self.local_conv = nn.Conv2d(ch, ch, 3, 1, 1)
        
        # 2. 统计特征提取：引入 3x3 和 5x5 的多尺度标准差感知
        self.std_s3 = WindowStd_Enhanced(3, ch)
        self.std_s5 = WindowStd_Enhanced(5, ch)
        
        # 3. 几何约束分支：通过 1x1 卷积压缩统计通道，提取显著性
        self.geo_compress = nn.Conv2d(ch * 2, ch, 1, bias=False)
        
        # 4. 全局校准分支：感知全局上下文以修正权重
        self.global_gap = nn.AdaptiveAvgPool2d(1)
        self.global_calib = nn.Sequential(
            nn.Conv2d(ch, ch // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // 4, ch, 1),
            nn.Sigmoid()
        )
        
        # 5. 权重生成层
        self.final_weight = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1, groups=ch), # 深度卷积增强空间感知
            nn.Sigmoid()
        )

    def forward(self, x):
        # 基础特征提取
        feat_base = self.local_conv(x)
        
        # 获取多尺度偏差统计 (Statistical Characteristics)
        s3 = self.std_s3(x)
        s5 = self.std_s5(x)
        
        # 融合多尺度统计信息 (Geometry-Aware Fusion)
        stat_combined = torch.cat([s3, s5], dim=1)
        stat_feat = self.geo_compress(stat_combined)
        
        # 计算全局校准权重 (Global Context Calibration)
        g_weight = self.global_calib(self.global_gap(x))
        
        # 生成最终调制权重：融合局部统计特征与全局偏置
        m_weight = self.final_weight(stat_feat * g_weight)
        
        # 执行加权调制
        out = feat_base * m_weight
        
        return out # 哔哩哔哩/微信公众号: CV缝合救星, 独家整理!

# 使用示例
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 模拟高分辨率图像特征输入 (B, C, H, W)
    input_tensor = torch.randn(1, 64, 128, 128).to(device)

    # 初始化 AS-GWM 模块
    model = AS_GWM(64).to(device)
    print(model)
    
    # 前向传播
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("--- AS-GWM 模块维度验证 ---")
    print("输入维度  :", input_tensor.shape)   
    print("输出维度  :", output_tensor.shape) 
    print("\n[创新成功]: 自适应统计-几何加权调制器已就绪。 \n")