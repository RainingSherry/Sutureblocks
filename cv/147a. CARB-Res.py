import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =========================
# 1. RMSNorm：Transformer中常用的归一化
# =========================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        # 计算最后一个维度上的均方根倒数
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


# =========================
# 2. CARB-Res 核心聚合器
#    Context-Adaptive Reweighted Block Residual Attention
# =========================
class CARBResAggregator(nn.Module):
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        self.dim = dim

        # -------------------------
        # 静态查询：类似论文里每层一个可学习查询向量
        # 这里直接定义成可学习参数
        # -------------------------
        self.static_query = nn.Parameter(torch.randn(dim) * 0.02)

        # -------------------------
        # 动态查询生成器：
        # 根据当前 partial_block 的全局上下文生成动态查询
        # 让查询不仅是固定参数，还能感知当前输入内容
        # -------------------------
        self.dynamic_query_proj = nn.Linear(dim, dim, bias=False)

        # -------------------------
        # 源特征可靠性打分：
        # 对每个候选源（历史 block + 当前 partial_block）估计一个可靠性 gate
        # 输出为标量，后续经过 sigmoid 转成 0~1 权重
        # -------------------------
        self.source_gate_proj = nn.Linear(dim, 1, bias=False)

        # -------------------------
        # 温度预测器：
        # 根据当前输入上下文预测 softmax 温度
        # 温度越低，选择越尖锐；温度越高，分布越平滑
        # -------------------------
        self.temperature_proj = nn.Linear(dim, 1, bias=True)

        # -------------------------
        # 通道重标定：
        # 对聚合结果做通道级重加权，增强重要通道
        # 这种写法很像 SE / 轻量通道注意力
        # -------------------------
        hidden_dim = max(dim // reduction, 8)
        self.channel_reweight = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, dim, bias=False),
            nn.Sigmoid()
        )

        # -------------------------
        # 用于 recency bias 的可学习强度参数
        # 值越大，越偏向更近的块
        # 用 softplus 保证它是正数
        # -------------------------
        self.recency_scale = nn.Parameter(torch.tensor(0.1))

        # -------------------------
        # 对候选源做 RMSNorm，避免某些块幅值过大导致注意力偏置
        # -------------------------
        self.norm = RMSNorm(dim)

    def forward(self, blocks: list[Tensor], partial_block: Tensor) -> tuple[Tensor, Tensor]:
        """
        参数说明
        ----------
        blocks:
            历史已经完成的 block 列表
            列表中每个张量形状为 [B, T, D]

        partial_block:
            当前 block 内部的部分累加结果
            形状为 [B, T, D]

        返回
        ----------
        h:
            聚合后的输出特征，形状 [B, T, D]

        attn_weights:
            在“源维度”上的注意力权重，形状 [N+1, B, T]
            其中 N 是历史 block 数量，最后一个源对应当前 partial_block
        """

        # -------------------------------------------------------
        # 1）构建候选源：
        #    历史 block + 当前 partial_block
        #    形成 [源数, B, T, D]
        # -------------------------------------------------------
        sources = blocks + [partial_block]
        V = torch.stack(sources, dim=0)  # [N+1, B, T, D]

        # 对候选源做归一化，作为 Key 特征
        K = self.norm(V)  # [N+1, B, T, D]

        # -------------------------------------------------------
        # 2）基于当前 partial_block 生成上下文向量
        #    这里用序列维平均池化得到全局上下文
        # -------------------------------------------------------
        context = partial_block.mean(dim=1)      # [B, D]
        context = self.norm(context)             # [B, D]

        # -------------------------------------------------------
        # 3）生成查询向量 q
        #    q = 静态查询 + 动态查询
        #    最后扩展到每个 token 上
        # -------------------------------------------------------
        static_q = self.static_query.view(1, 1, self.dim)                 # [1, 1, D]
        dynamic_q = self.dynamic_query_proj(context).unsqueeze(1)         # [B, 1, D]
        q = static_q + dynamic_q                                          # [B, 1, D]
        q = q.expand(-1, partial_block.size(1), -1)                       # [B, T, D]

        # -------------------------------------------------------
        # 4）计算基础相似度分数
        #    对每个源、每个 batch、每个 token 做点积
        # -------------------------------------------------------
        raw_logits = torch.einsum('btd,nbtd->nbt', q, K)  # [N+1, B, T]

        # -------------------------------------------------------
        # 5）计算源可靠性 gate
        #    每个候选源都有一个内容相关的可靠性分数
        # -------------------------------------------------------
        source_gate = torch.sigmoid(self.source_gate_proj(K).squeeze(-1))  # [N+1, B, T]

        # 为了数值稳定，将 gate 转到对数域，与 logits 相加
        gate_bias = torch.log(source_gate + 1e-6)  # [N+1, B, T]

        # -------------------------------------------------------
        # 6）计算 recency bias（近邻偏置）
        #    越靠近当前层的源，偏置越大
        #    这有助于保留局部连续建模能力
        # -------------------------------------------------------
        num_sources = V.size(0)
        device = V.device

        # 源索引：0, 1, 2, ..., N
        source_index = torch.arange(num_sources, device=device, dtype=V.dtype)

        # 当前 partial_block 在最后一个位置，因此离当前越近，距离越小
        distance = (num_sources - 1) - source_index  # [N+1]
        recency_strength = F.softplus(self.recency_scale)  # 保证为正
        recency_bias = -recency_strength * distance        # [N+1]
        recency_bias = recency_bias.view(num_sources, 1, 1)  # [N+1, 1, 1]

        # -------------------------------------------------------
        # 7）动态温度
        #    让不同样本自适应决定注意力的“尖锐程度”
        # -------------------------------------------------------
        temperature = F.softplus(self.temperature_proj(context)) + 0.5  # [B, 1]
        temperature = temperature.transpose(0, 1).unsqueeze(-1)         # [1, B, 1]

        # -------------------------------------------------------
        # 8）融合所有偏置项，计算最终注意力
        # -------------------------------------------------------
        logits = raw_logits / temperature + gate_bias + recency_bias    # [N+1, B, T]
        attn_weights = torch.softmax(logits, dim=0)                     # 在源维度做 softmax

        # -------------------------------------------------------
        # 9）用注意力权重聚合源特征
        # -------------------------------------------------------
        h = torch.einsum('nbt,nbtd->btd', attn_weights, V)              # [B, T, D]

        # -------------------------------------------------------
        # 10）做一次通道重标定
        #     相当于对聚合结果进行细化增强
        # -------------------------------------------------------
        channel_gate = self.channel_reweight(h.mean(dim=1))             # [B, D]
        h = h * (1.0 + channel_gate.unsqueeze(1))                       # [B, T, D]

        return h, attn_weights


# =========================
# 3. 改进版 Transformer Layer
#    使用 CARB-Res 分别替代原来的 attn residual 与 mlp residual
# =========================
class CARBResTransformerLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        layer_number: int,
        block_size: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.layer_number = layer_number
        self.block_size = block_size

        # -------------------------
        # 在 Attention 前使用一个 CARB-Res 聚合器
        # -------------------------
        self.pre_attn_res = CARBResAggregator(dim)

        # -------------------------
        # 在 MLP 前使用另一个 CARB-Res 聚合器
        # 这样两处可以学习到不同的跨层聚合策略
        # -------------------------
        self.pre_mlp_res = CARBResAggregator(dim)

        # -------------------------
        # 注意力分支
        # -------------------------
        self.attn_norm = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # -------------------------
        # MLP 分支
        # -------------------------
        self.mlp_norm = RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, blocks: list[Tensor], hidden_states: Tensor):
        """
        参数
        ----------
        blocks:
            历史已完成块列表，每个元素形状 [B, T, D]

        hidden_states:
            当前输入，形状 [B, T, D]

        返回
        ----------
        output_blocks:
            更新后的历史块列表

        output_partial:
            当前 block 的部分和

        aux_info:
            附加信息字典，可用于可视化或调试
        """

        # 为避免原地修改外部输入，这里复制一份列表
        blocks = list(blocks)

        # -------------------------------------------------------
        # 当前 block 内部的部分和，初始化为输入 hidden_states
        # -------------------------------------------------------
        partial_block = hidden_states

        # -------------------------------------------------------
        # 1）在 self-attention 前做 CARB-Res 聚合
        # -------------------------------------------------------
        h_attn_in, attn_weights_1 = self.pre_attn_res(blocks, partial_block)

        # -------------------------------------------------------
        # 2）判断是否到达 block 边界
        #    注意：原始代码中 block_size 统计的是 “ATTN + MLP”
        #    一个 Transformer layer 里有 2 个子层，因此这里除以 2
        # -------------------------------------------------------
        if self.layer_number % (self.block_size // 2) == 0:
            # 到达边界时，把当前 partial_block 作为已完成块存起来
            blocks.append(partial_block)
            partial_block = None

        # -------------------------------------------------------
        # 3）执行自注意力
        # -------------------------------------------------------
        h_norm = self.attn_norm(h_attn_in)
        attn_out = self.attn(h_norm, h_norm, h_norm, need_weights=False)[0]

        # 如果已经切块，则当前 partial_block 重新从 attn_out 开始
        # 否则累加到当前 partial_block 中
        partial_block = attn_out if partial_block is None else (partial_block + attn_out)

        # -------------------------------------------------------
        # 4）在 MLP 前再次做 CARB-Res 聚合
        # -------------------------------------------------------
        h_mlp_in, attn_weights_2 = self.pre_mlp_res(blocks, partial_block)

        # -------------------------------------------------------
        # 5）执行 MLP
        # -------------------------------------------------------
        mlp_out = self.mlp(self.mlp_norm(h_mlp_in))
        partial_block = partial_block + mlp_out

        # -------------------------------------------------------
        # 6）附加调试信息
        # -------------------------------------------------------
        aux_info = {
            "pre_attn_weights": attn_weights_1,   # [N+1, B, T]
            "pre_mlp_weights": attn_weights_2     # [N+1, B, T]
        }

        return blocks, partial_block, aux_info


# =========================
# 4. 小测试：直接运行验证
# =========================
if __name__ == "__main__":
    # 固定随机种子，便于复现
    torch.manual_seed(42)

    # -------------------------
    # 输入维度设置
    # -------------------------
    batch_size = 2       # B：批次大小
    seq_len = 10         # T：序列长度
    hidden_dim = 64      # D：特征维度
    num_heads = 8        # 多头注意力头数
    block_size = 4       # 一个 block 内统计的子层数（ATTN + MLP）
    layer_number = 2     # 当前层编号

    # -------------------------
    # 初始化模型层
    # -------------------------
    model_layer = CARBResTransformerLayer(
        dim=hidden_dim,
        num_heads=num_heads,
        layer_number=layer_number,
        block_size=block_size,
        mlp_ratio=4,
        dropout=0.0
    )

    print("========== 模块名称 ==========")
    print("CARB-Res 模块（Context-Adaptive Reweighted Block Residual Attention）")
    print()

    print("========== 模型结构 ==========")
    print(model_layer)
    print()

    # -------------------------
    # 构造历史 block
    # 假设已经有 2 个历史 block
    # 每个 block 形状为 [B, T, D]
    # -------------------------
    blocks = [
        torch.randn(batch_size, seq_len, hidden_dim),
        torch.randn(batch_size, seq_len, hidden_dim)
    ]

    # 当前层输入 hidden_states
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)

    # -------------------------
    # 前向传播
    # -------------------------
    model_layer.eval()
    with torch.no_grad():
        output_blocks, output_partial, aux_info = model_layer(blocks, hidden_states)

    # -------------------------
    # 打印结果
    # -------------------------
    print("========== 输入输出信息 ==========")
    print(f"输入 hidden_states 形状: {hidden_states.shape}")
    print(f"输出 partial_block 形状: {output_partial.shape}")
    print(f"输出 blocks 数量: {len(output_blocks)}")
    print(f"pre-attn 注意力权重形状: {aux_info['pre_attn_weights'].shape}")
    print(f"pre-mlp 注意力权重形状: {aux_info['pre_mlp_weights'].shape}")
    print()

    # 简单检查数值是否正常
    print("========== 数值统计 ==========")
    print(f"输出均值: {output_partial.mean().item():.6f}")
    print(f"输出标准差: {output_partial.std().item():.6f}")
    print()

    print("运行完成，CARB-Res 模块前向传播正常。")