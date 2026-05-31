import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ============================================================
# 工具函数：对最后一个维度做安全归一化
# ============================================================
def safe_l1_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """
    对张量在指定维度做 L1 归一化，避免分母为 0。
    参数:
        x: 输入张量
        dim: 归一化维度
        eps: 数值稳定项
    返回:
        归一化后的张量
    """
    return x / (x.sum(dim=dim, keepdim=True) + eps)


def safe_frobenius_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    对张量最后两个维度做 Frobenius 归一化。
    这是原始 InfSA 中非常关键的一步，用于控制注意力矩阵整体范数，
    让无限传播级数更加稳定。

    参数:
        x: 输入张量，形状通常为 (..., N, S)
        eps: 数值稳定项
    返回:
        Frobenius 归一化后的张量
    """
    original_shape = x.shape
    x_flat = x.flatten(-2)                          # (..., N*S)
    frob_norm = torch.norm(x_flat, p=2, dim=-1, keepdim=True)  # (..., 1)
    x_flat = x_flat / (frob_norm + eps)
    return x_flat.view(original_shape)


# ============================================================
# 原始 Pure InfSA 分数计算
# ============================================================
def pure_infsa_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    rho: Union[float, torch.Tensor] = 0.95,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    原始 Pure InfSA 注意力分数计算。

    核心思想：
        A = rho * FroNorm( ReLU(QK^T / sqrt(D)) )

    这里没有 Softmax，而是：
    1. 先做缩放点积
    2. 再做 ReLU，只保留正相关关系
    3. 再做 Frobenius 归一化，保证整体传播稳定
    4. 最后乘可学习或固定的 rho 控制传播强度

    参数:
        q: Query，形状 (B, H, N, D)
        k: Key，形状 (B, H, S, D)
        rho: 衰减系数，可为 float 或 Tensor
        eps: 数值稳定项
    返回:
        注意力矩阵，形状 (B, H, N, S)
    """
    d = q.shape[-1]
    scale = math.sqrt(1.0 / d)

    q_scaled = q * scale
    k_scaled = k * scale

    # 计算原始相关矩阵
    attn = torch.matmul(q_scaled, k_scaled.transpose(-2, -1))  # (B, H, N, S)

    # 仅保留正相关边
    attn = torch.relu(attn)

    # Frobenius 归一化，控制整体传播强度
    attn = safe_frobenius_normalize(attn, eps=eps)

    return rho * attn


# ============================================================
# 原始 Linear InfSA 分数计算
# ============================================================
def linear_infsa_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    rho: Union[float, torch.Tensor] = 0.95,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    原始 Linear InfSA 的线性近似版本。

    核心思想：
    不显式构造 N×N 注意力矩阵，而是：
    1. 先从 Query 中提取一个全局能量加权摘要 q_bar
    2. 再用 q_bar 和全部 Key 做相关性计算
    3. 得到每个 token 的全局重要性分数
    4. 后续用它来对 V 做加权池化

    这样复杂度从 O(N^2) 降到近似 O(ND)。

    参数:
        q: Query，形状 (B, H, N, D)
        k: Key，形状 (B, H, S, D)
        rho: 衰减系数
        eps: 数值稳定项
    返回:
        分数向量，形状 (B, H, S, 1)
    """
    d = q.shape[-1]
    scale = math.sqrt(1.0 / d)

    q_scaled = q * scale
    k_scaled = k * scale

    # 第一步：基于 Query 能量生成全局摘要 q_bar
    energy = torch.relu(q_scaled.norm(p=2, dim=-1))          # (B, H, N)
    weights = safe_l1_normalize(energy, dim=-1, eps=eps)     # (B, H, N)
    q_bar = torch.einsum("bhn,bhnd->bhd", weights, q_scaled) # (B, H, D)

    # 第二步：用全局摘要 q_bar 与 Key 计算每个 token 的重要性
    scores = torch.relu(torch.einsum("bhd,bhsd->bhs", q_bar, k_scaled))  # (B, H, S)
    scores = safe_l1_normalize(scores, dim=-1, eps=eps)

    return (rho * scores).unsqueeze(-1)  # (B, H, S, 1)


# ============================================================
# 新增：SCG-InfSA 分数计算
# ============================================================
def scg_infsa_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    rho: Union[float, torch.Tensor] = 0.95,
    saliency_alpha: float = 1.0,
    centrality_beta: float = 0.5,
    self_loop_gamma: float = 0.1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    SCG-InfSA：Saliency-Centrality Gated Infinite Self-Attention
    显著性-中心性门控无限自注意力

    这是在原始 Pure InfSA 基础上的一个“CVPR 风格魔改”：
    ------------------------------------------------------------
    1. 先保留原始 InfSA 的正相关传播骨架：
       base = ReLU(QK^T / sqrt(D))

    2. 再引入“显著性门控”：
       - 对 Query token 和 Key token 分别估计显著性
       - 用外积构造 pair-wise 的边增强图
       - 让显著 token 之间的传播边被强化
       直观上：让模型更关注“像目标”的 token，而不是平均传播

    3. 再引入“中心性增强”：
       - 将 base 图看成一个有向图
       - 统计每个 query 节点的出边强度、每个 key 节点的入边强度
       - 构造 query-key 双侧中心性增强项
       直观上：重要节点之间的连接应该更强

    4. 再引入“自环保真残差”：
       - 当 N == S 时，额外给对角线一部分增强
       - 保留 token 自身身份信息，减轻无限传播时的过平滑

    5. 最后重新做 Frobenius 归一化，并乘 rho
       - 保证数值稳定
       - 保持和原始 InfSA 一致的传播风格

    参数:
        q: Query，形状 (B, H, N, D)
        k: Key，形状 (B, H, S, D)
        rho: 衰减系数
        saliency_alpha: 显著性门控强度
        centrality_beta: 中心性增强强度
        self_loop_gamma: 自环增强强度
        eps: 数值稳定项
    返回:
        注意力矩阵，形状 (B, H, N, S)
    """
    d = q.shape[-1]
    scale = math.sqrt(1.0 / d)

    q_scaled = q * scale
    k_scaled = k * scale

    # --------------------------------------------------------
    # 第 1 步：构造原始相关图
    # --------------------------------------------------------
    base = torch.matmul(q_scaled, k_scaled.transpose(-2, -1))  # (B, H, N, S)
    base = torch.relu(base)

    # --------------------------------------------------------
    # 第 2 步：显著性门控
    # --------------------------------------------------------
    # 分别计算 Query 与 Key token 的能量，作为显著性估计
    q_energy = torch.relu(q_scaled.norm(p=2, dim=-1))  # (B, H, N)
    k_energy = torch.relu(k_scaled.norm(p=2, dim=-1))  # (B, H, S)

    # 做归一化，避免数值漂移
    q_sal = safe_l1_normalize(q_energy, dim=-1, eps=eps)  # (B, H, N)
    k_sal = safe_l1_normalize(k_energy, dim=-1, eps=eps)  # (B, H, S)

    # 构造成 pair-wise 显著性外积
    # 这一步会形成一个边级别的“显著连接增强图”
    saliency_gate = torch.einsum("bhn,bhs->bhns", q_sal, k_sal)  # (B, H, N, S)

    # 用 1 + alpha * gate 的方式做乘性增强
    # 这样不会破坏原始边，只会对重要边做额外放大
    attn = base * (1.0 + saliency_alpha * saliency_gate)

    # --------------------------------------------------------
    # 第 3 步：中心性增强
    # --------------------------------------------------------
    # query 中心性：每个 query 节点向外连接的总强度
    q_centrality = base.sum(dim=-1)  # (B, H, N)

    # key 中心性：每个 key 节点被连接的总强度
    k_centrality = base.sum(dim=-2)  # (B, H, S)

    # 各自归一化
    q_centrality = safe_l1_normalize(q_centrality, dim=-1, eps=eps)
    k_centrality = safe_l1_normalize(k_centrality, dim=-1, eps=eps)

    # 构造成 query-key 双侧中心性增强图
    centrality_gate = torch.einsum("bhn,bhs->bhns", q_centrality, k_centrality)

    # 做加性增强，让图中“重要节点之间”的连接更容易凸显
    attn = attn + centrality_beta * centrality_gate

    # --------------------------------------------------------
    # 第 4 步：自环保真残差
    # --------------------------------------------------------
    # 当是标准自注意力时，N == S，此时可以给对角线额外增强
    # 目的是保留 token 自身身份，减轻传播过程中被过度平滑
    n = q.shape[-2]
    s = k.shape[-2]
    if n == s and self_loop_gamma > 0:
        eye = torch.eye(n, device=q.device, dtype=q.dtype).view(1, 1, n, n)
        attn = attn + self_loop_gamma * eye

    # --------------------------------------------------------
    # 第 5 步：重新做 Frobenius 归一化并乘 rho
    # --------------------------------------------------------
    attn = safe_frobenius_normalize(attn, eps=eps)
    attn = rho * attn

    return attn


# ============================================================
# 主注意力计算函数
# ============================================================
def infsa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    variant: str = "pure_infsa",
    rho: Union[float, torch.Tensor] = 0.95,
    dropout_p: float = 0.0,
    training: bool = False,
    eps: float = 1e-6,
    saliency_alpha: float = 1.0,
    centrality_beta: float = 0.5,
    self_loop_gamma: float = 0.1,
    return_scores: bool = False,
):
    """
    主函数：根据不同 variant 计算注意力输出。

    支持三种模式：
    1. pure_infsa   : 原始全矩阵 InfSA
    2. linear_infsa : 原始线性近似 InfSA
    3. scg_infsa    : 新增的显著性-中心性门控版本

    参数:
        q, k, v: 投影后的多头张量，形状均为 (B, H, N/S, D)
        variant: 注意力变体名称
        rho: 衰减系数
        dropout_p: 分数上的 dropout
        training: 是否训练模式
        eps: 数值稳定项
        saliency_alpha: 仅用于 scg_infsa，显著性门控强度
        centrality_beta: 仅用于 scg_infsa，中心性增强强度
        self_loop_gamma: 仅用于 scg_infsa，自环增强强度
        return_scores: 是否返回注意力分数
    返回:
        output 或 (output, scores)
    """
    if variant == "pure_infsa":
        scores = pure_infsa_scores(q, k, rho=rho, eps=eps)   # (B, H, N, S)
        if dropout_p > 0.0 and training:
            scores = F.dropout(scores, p=dropout_p)
        output = torch.matmul(scores, v)                     # (B, H, N, D)

    elif variant == "linear_infsa":
        scores = linear_infsa_scores(q, k, rho=rho, eps=eps)  # (B, H, S, 1)
        if dropout_p > 0.0 and training:
            scores = F.dropout(scores, p=dropout_p)

        # 线性近似版本中，不再有完整的 N×S 注意力矩阵
        # 而是先对所有 value 做一次全局加权池化，再广播回每个 query 位置
        pooled = (scores * v).sum(dim=-2, keepdim=True)      # (B, H, 1, D)
        n = q.shape[-2]
        output = pooled.expand(q.shape[0], q.shape[1], n, q.shape[-1])  # (B, H, N, D)

    elif variant == "scg_infsa":
        scores = scg_infsa_scores(
            q=q,
            k=k,
            rho=rho,
            saliency_alpha=saliency_alpha,
            centrality_beta=centrality_beta,
            self_loop_gamma=self_loop_gamma,
            eps=eps,
        )  # (B, H, N, S)

        if dropout_p > 0.0 and training:
            scores = F.dropout(scores, p=dropout_p)

        output = torch.matmul(scores, v)  # (B, H, N, D)

    else:
        raise ValueError(
            f"未知的 variant: {variant}，可选为 "
            f"'pure_infsa'、'linear_infsa'、'scg_infsa'"
        )

    if return_scores:
        return output, scores
    return output


# ============================================================
# 多头注意力模块封装
# ============================================================
class InfSAAttention(nn.Module):
    """
    多头 InfSA / SCG-InfSA 模块

    这是一个可以直接替换标准多头注意力的模块。
    相比你原始版本，这里有几个增强点：

    1. 保留了原始 pure_infsa 和 linear_infsa
    2. 新增 scg_infsa 版本
    3. 修复了原代码中 rho 用 .item() 导致无法学习的问题
    4. 支持 need_weights=True 时返回注意力分数
    5. 注释全部中文，便于你后续继续改论文图和代码

    支持输入格式：
        - batch_first=True  : (B, N, E)
        - batch_first=False : (N, B, E)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        variant: str = "scg_infsa",
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
        rho_init: float = 0.95,
        rho_trainable: bool = True,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        saliency_alpha: float = 1.0,
        centrality_beta: float = 0.5,
        self_loop_gamma: float = 0.1,
    ):
        super().__init__()

        if variant not in ("pure_infsa", "linear_infsa", "scg_infsa"):
            raise ValueError(
                f"未知的 variant: {variant}，可选为 "
                f"'pure_infsa'、'linear_infsa'、'scg_infsa'"
            )

        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError("embed_dim 和 num_heads 必须大于 0")

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim 必须能被 num_heads 整除")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.variant = variant
        self.dropout = dropout
        self.batch_first = batch_first

        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        # ----------------------------------------------------
        # 这三个参数仅用于 SCG-InfSA
        # ----------------------------------------------------
        self.saliency_alpha = saliency_alpha
        self.centrality_beta = centrality_beta
        self.self_loop_gamma = self_loop_gamma

        # ----------------------------------------------------
        # 线性投影层：将输入映射到 Q、K、V
        # ----------------------------------------------------
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # ----------------------------------------------------
        # 可学习 rho：使用 logit 参数化，保证 rho 始终位于 (0, 1)
        # rho = sigmoid(rho_logit)
        # ----------------------------------------------------
        rho_init = float(rho_init)
        rho_init = min(max(rho_init, 1e-4), 1 - 1e-4)
        rho_logit = math.log(rho_init / (1.0 - rho_init))

        if rho_trainable:
            self.rho_logit = nn.Parameter(torch.tensor(rho_logit, dtype=torch.float32))
        else:
            self.register_buffer("rho_logit", torch.tensor(rho_logit, dtype=torch.float32))

        self._reset_parameters()

    @property
    def rho(self) -> float:
        """
        返回当前 rho 的浮点值，便于打印观察。
        注意：训练中实际计算不要用这个属性，因为它会转成 Python float。
        """
        return torch.sigmoid(self.rho_logit).detach().item()

    def _reset_parameters(self):
        """
        参数初始化。
        使用 Xavier 初始化线性层权重，偏置清零。
        """
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        query: Optional[Tensor] = None,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
        **kwargs,
    ):
        """
        前向传播。

        参数说明：
            query, key, value:
                输入张量。若只传 query，则默认做自注意力，即 key=value=query。

            key_padding_mask:
                当前版本仅为接口兼容保留，未实际使用。

            need_weights:
                若为 True，则额外返回注意力分数。

            attn_mask:
                当前版本未使用，仅为兼容常见注意力接口。

            average_attn_weights:
                当 need_weights=True 时，是否在多头维度做平均。

            is_causal:
                当前版本未实现因果掩码，仅接口兼容。

            kwargs:
                额外兼容 hidden_states 写法，便于和部分 Transformer 框架对接。

        返回：
            - 若 need_weights=False：返回 output
            - 若 need_weights=True ：返回 (output, attn_weights)
        """
        # ----------------------------------------------------
        # 兼容部分框架直接传 hidden_states 的写法
        # ----------------------------------------------------
        hidden_states = kwargs.get("hidden_states", None)
        if hidden_states is not None and query is None:
            query = key = value = hidden_states

        # ----------------------------------------------------
        # 若只传 query，则自动补成标准自注意力
        # ----------------------------------------------------
        if query is not None:
            if key is None:
                key = query
            if value is None:
                value = key

        # ----------------------------------------------------
        # 安全校验
        # ----------------------------------------------------
        if query is None or key is None or value is None:
            raise ValueError("query、key、value 不能为空")

        # ----------------------------------------------------
        # 若输入不是 batch_first，则转成 (B, N, E)
        # ----------------------------------------------------
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        # ----------------------------------------------------
        # 读取尺寸信息
        # ----------------------------------------------------
        b, n, _ = query.shape
        s = key.shape[1]
        h = self.num_heads
        d = self.head_dim

        # ----------------------------------------------------
        # 线性投影并拆分成多头
        # 形状：
        #   q: (B, H, N, D)
        #   k: (B, H, S, D)
        #   v: (B, H, S, D)
        # ----------------------------------------------------
        q = self.q_proj(query).view(b, n, h, d).transpose(1, 2)
        k = self.k_proj(key).view(b, s, h, d).transpose(1, 2)
        v = self.v_proj(value).view(b, s, h, d).transpose(1, 2)

        # ----------------------------------------------------
        # 这里非常重要：
        # 不再使用 .item()，避免切断 rho 的梯度
        # ----------------------------------------------------
        rho_tensor = torch.sigmoid(self.rho_logit).to(q.dtype).to(q.device)

        # ----------------------------------------------------
        # 调用主注意力函数
        # ----------------------------------------------------
        if need_weights:
            output, attn_weights = infsa_attention(
                q=q,
                k=k,
                v=v,
                variant=self.variant,
                rho=rho_tensor,
                dropout_p=self.dropout if self.training else 0.0,
                training=self.training,
                saliency_alpha=self.saliency_alpha,
                centrality_beta=self.centrality_beta,
                self_loop_gamma=self.self_loop_gamma,
                return_scores=True,
            )
        else:
            output = infsa_attention(
                q=q,
                k=k,
                v=v,
                variant=self.variant,
                rho=rho_tensor,
                dropout_p=self.dropout if self.training else 0.0,
                training=self.training,
                saliency_alpha=self.saliency_alpha,
                centrality_beta=self.centrality_beta,
                self_loop_gamma=self.self_loop_gamma,
                return_scores=False,
            )
            attn_weights = None

        # ----------------------------------------------------
        # 多头结果拼回原始嵌入维度
        # (B, H, N, D) -> (B, N, E)
        # ----------------------------------------------------
        output = output.transpose(1, 2).contiguous().view(b, n, self.embed_dim)

        # 输出投影
        output = self.out_proj(output)

        # ----------------------------------------------------
        # 若原始输入不是 batch_first，则转回去
        # ----------------------------------------------------
        if not self.batch_first:
            output = output.transpose(0, 1)

        # ----------------------------------------------------
        # 若需要返回注意力权重，则按要求处理
        # ----------------------------------------------------
        if need_weights:
            # 对 linear_infsa，返回的是 (B,H,S,1)，它本身不是标准矩阵注意力
            # pure_infsa / scg_infsa 返回的是 (B,H,N,S)
            if average_attn_weights and attn_weights is not None and attn_weights.dim() >= 4:
                attn_weights = attn_weights.mean(dim=1)  # 对 head 取平均 -> (B,N,S)
            return output, attn_weights

        return output


# ============================================================
# 一个最小可运行测试
# ============================================================
if __name__ == "__main__":
    """
    这个 main 部分可以直接运行，用于快速验证模块是否工作正常。
    你可以直接复制整份代码保存为 .py 文件运行。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------------
    # 构造一个模拟输入：
    # B=4, N=197, E=512
    # 这是 ViT 场景里很常见的 patch token 规模
    # --------------------------------------------------------
    input_tensor = torch.randn(4, 197, 512).to(device)

    # --------------------------------------------------------
    # 你可以在这里切换不同版本：
    # variant="pure_infsa"
    # variant="linear_infsa"
    # variant="scg_infsa"
    # --------------------------------------------------------
    model = InfSAAttention(
        embed_dim=512,
        num_heads=8,
        variant="scg_infsa",      # 新版魔改
        dropout=0.1,
        batch_first=True,
        rho_init=0.95,
        rho_trainable=True,
        saliency_alpha=2.0,       # 显著性门控强度
        centrality_beta=0.8,      # 中心性增强强度
        self_loop_gamma=0.15,     # 自环增强强度
    ).to(device)

    print("当前模型结构：")
    print(model)
    print()

    # 前向传播
    output_tensor, attn_weights = model(input_tensor, need_weights=True)

    # 打印结果形状
    print("输入张量形状 input_tensor.shape :", input_tensor.shape)
    print("输出张量形状 output_tensor.shape:", output_tensor.shape)

    if attn_weights is not None:
        print("注意力权重形状 attn_weights.shape:", attn_weights.shape)

    print(f"当前可学习 rho 值: {model.rho:.6f}")
    print("\n模块运行成功：SCG-InfSA（显著性-中心性门控无限自注意力）")