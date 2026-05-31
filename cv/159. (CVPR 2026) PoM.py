import torch
import torch.nn as nn

# =========================================================
# CV缝合救星独家复现：Polynomial Mixer (PoM) 核心模块
# =========================================================
class PoM(nn.Module):
    def __init__(self, dim, hidden_dim=None, k=2, act_layer=nn.GELU):
        """
        dim: 输入特征的通道数
        hidden_dim: 内部展开的隐藏层维度 D，论文里推荐设为 2 倍的 dim
        k: 多项式的阶数，默认 2 阶就够打了
        """
        super().__init__()
        # CV缝合救星提示：不传 hidden_dim 默认直接翻倍
        hidden_dim = hidden_dim or dim * 2
        self.k = k
        self.hidden_dim = hidden_dim

        # 对应论文中的 W_h，把特征投影到更高维的 D
        self.Wh = nn.Linear(dim, hidden_dim)
        self.act_h = act_layer()

        # 核心精髓！多项式系数 alpha，必须是可学习参数。大小为 [k, hidden_dim]
        # 初始化给点微小的扰动，方便梯度反传
        self.alpha = nn.Parameter(torch.randn(k, hidden_dim) * 0.02)

        # 对应论文中的 W_s，用来生成 gating (门控) 系数
        self.Ws = nn.Linear(dim, hidden_dim)
        self.act_s = nn.Sigmoid()

        # 对应论文中的 W_o，把特征映射回原来的通道数，深藏功与名
        self.Wo = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        # 进来的 x 形状必须是 (B, N, C)，N 是序列长度 (或者像素数量 H*W)
        B, N, C = x.shape

        # === 第一条路：计算共享状态 H(X) ===
        # 先过线性层和激活函数
        hx = self.act_h(self.Wh(x))  # 形状: [B, N, hidden_dim]

        # 算多项式，CV缝合救星老套路，用个循环搞定 k 阶求和
        poly_sum = torch.zeros_like(hx)
        for p in range(1, self.k + 1):
            # self.alpha[p-1] 形状 [hidden_dim]，乘的时候 PyTorch 会自动广播到 [B, N, hidden_dim]
            poly_sum += self.alpha[p-1] * (hx ** p)

        # 沿着序列长度 N 求和，把上下文压缩成一个紧凑的特征！(这就是不用算 N^2 注意力矩阵的秘密)
        H_X = poly_sum.sum(dim=1, keepdim=True)  # 形状压缩为: [B, 1, hidden_dim]

        # === 第二条路：计算门控查询 S(X) ===
        sx = self.act_s(self.Ws(x))  # 形状: [B, N, hidden_dim]

        # === 两路合并 ===
        # 用门控 sx 去检索全局状态 H_X，这里广播机制再次立功
        out = sx * H_X  # 形状恢复: [B, N, hidden_dim]

        # === 投影输出 ===
        out = self.Wo(out)  # 形状回归: [B, N, C]

        return out

# =========================================================
# CV缝合救星贴心加餐：给 CNN 和视觉任务准备的 2D Wrapper
# =========================================================
class PoM_2D(nn.Module):
    def __init__(self, dim, hidden_dim=None, k=2):
        super().__init__()
        self.pom = PoM(dim, hidden_dim, k)

    def forward(self, x):
        # 视觉任务标准的四维张量: [B, C, H, W]
        B, C, H, W = x.shape
        
        # CV缝合救星基操：把 2D 压平并转置，变成 [B, N, C] 给 PoM 吃
        x_flat = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        # 丢进主模块
        out = self.pom(x_flat)
        
        # 算完再转回来，缝衣针拔出，完美无痕
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return out


# 测试代码：跑跑看结构和维度对不对
if __name__ == "__main__":
    # 模拟一个 Batch Size = 2, 通道数 = 64, 高宽 = 32x32 的特征图输入
    dummy_input = torch.randn(2, 64, 32, 32)
    
    print("🚀 CV缝合救星启动：开始测试 PoM_2D 模块...")
    
    # 实例化模型
    model = PoM_2D(dim=64, hidden_dim=128, k=2)
    
    # 打印网络结构看看
    print("\n--- 打印模型结构 ---")
    print(model)
    
    # 前向传播
    output = model(dummy_input)
    
    print("\n--- 维度检查 (CV缝合救星保底认证) ---")
    print(f"输入 Tensor 形状: {dummy_input.shape}")
    print(f"输出 Tensor 形状: {output.shape}")
    
    if dummy_input.shape == output.shape:
        print("✅ 缝合成功！输入输出维度完全一致，快拿去发 paper 吧！")