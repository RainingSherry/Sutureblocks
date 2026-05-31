# 缝合模块创新点地图

> 本文件按**创新策略**对所有模块进行归类，帮助 Agent 快速找到创新组合思路。
> 数据来源：`scripts/query.py` + `index.db`

---

## 创新策略总览

| 策略 | 核心思路 | 代表模块 |
|------|----------|----------|
| **注意力机制改进** | 降低复杂度 / 增强稀疏性 | BiFormer, Agent-Attention, EfficientAttention |
| **频域建模** | 在 FFT/DWT/小波域处理 | FreqFusion, FourierUnit, Wavelet |
| **卷积形态演进** | 深度可分离 / 可变形 / 条带 | PConv, DCNv2/v3, StripPooling |
| **门控机制** | 动态过滤无用信息 | GLU, CGLU, SwiGLU, Gated |
| **Mamba/状态空间** | 线性复杂度的长程建模 | CSMamba, VMamba, MLLAttention |
| **归一化改进** | 更稳定的学习信号 | RMSNorm, LayerNorm, GroupNorm |
| **多尺度融合** | 跨分辨率特征聚合 | FPN, ASPP, PPM, SAFM |
| **位置编码增强** | 补充绝对/相对位置信息 | RoPE, CoTAttention, CoordAttention |
| **混合架构** | CNN+Transformer / 卷积+注意力 | ACmix, CoAtNet, CBAM |
| **通道/空间分离** | 分别建模不同维度 | SE, CBAM, SimAM, SGE |
| **稀疏注意力** | Top-K / 窗口 / 条带 | WindowAttention, HaloAttention, SSA |
| **特征融合** | 动态加权组合多分支 | FreqFusion, SCA, EMA |
| **残差/跳跃设计** | 保留原始信号 | 大量模块的标配设计 |

---

## 详细创新点分类

### 1. 注意力机制改进

#### 1.1 稀疏路由注意力（Bi-level Routing）
- **原理**：先在区域级别筛选相关区域，再在 token 级别做精细注意力
- **代表**：`BiFormer (CVPR 2023)` → `BiLevelRoutingAttention_nchw`
- **创新点**：O(n) 复杂度 + 动态 topk 筛选，保留关键信息
- **缝合思路**：把 BiFormer 的路由机制替换进其他 ViT backbone

#### 1.2 代理令牌注意力（Agent Attention）
- **原理**：引入 Agent tokens 作为 Q 和 KV 之间的信息中介
- **代表**：`Agent-Attention (ECCV 2024)` → `AgentAttention(dim, agent_num=49)`
- **创新点**：两步聚合：Agent ← KV → Q，融合 Softmax 和 Linear attention 的优势
- **缝合思路**：替换 ViT / Swin-UNet 中的 WindowAttention

#### 1.3 差分注意力（Differential Attention）
- **原理**：两个 Softmax 分布相减，消除共同噪声
- **代表**：`MHDA (Arxiv 2024)` → `MultiHeadDifferentialAttention`
- **创新点**：λ 参数可学习，像降噪耳机一样去除注意力噪声
- **缝合思路**：替换 Transformer encoder 中的 MultiHeadAttention

#### 1.4 高效注意力（Linear/Softmax 近似）
- **原理**：用核函数近似 Softmax(QK^T) 或直接用线性复杂度的方案
- **代表**：`EfficientAttention (Arxiv 2024)`、`Sea_Attention (ICLR 2023)`
- **创新点**：去除 Softmax 的指数爆炸，支持高分辨率输入
- **缝合思路**：图像超分 / 去噪网络中替换标准 self-attention

#### 1.5 条带/窗口注意力（Strip / Window Attention）
- **原理**：限制注意力范围到条带或窗口内，降低复杂度
- **代表**：`SSA (NN 2024)` → `SSA(dim, group, kernel)`、`HaloAttention (ICCV 2021)`
- **创新点**：条带注意力的水平和垂直方向分离，并行度高
- **缝合思路**：图像恢复任务中替换全局注意力

#### 1.6 通道维度注意力（Transposed Attention）
- **原理**：在通道维度而非空间维度做注意力
- **代表**：`Restormer (CVPR 2022)` → `MultiDconvHeadTransposedAttention`
- **创新点**：O(H×W) 复杂度，处理高分辨率特征图
- **缝合思路**：low-level 视觉任务（去噪、去雾等）

#### 1.7 即插即用注意力全家桶
- **CVPR 系列**：`ASSA (CVPR 2024)`、`CAA (CVPR 2024)`、`SHSA`、`Manhattan_Self_Attention`
- **ICCV 系列**：`HaloAttention`、`ResidualAttention`、`DySample`
- **ECCV 系列**：`Agent-Attention`、`DHSA`
- **NeurIPS/ICLR**：`CoAtNet`、`Sea_Attention`、`ODConv`、`MobileViTAttention`
- **AAAI 系列**：`ScaleGraphBlock`、`FCM和MKP`、`DTAB和GCSA`、`SSA稀疏自注意力`
- **混合注意力**：`CoTAttention`（上下文 + 坐标）、`TripletAttention`（通道 + 空间 + 极坐标）

---

### 2. 频域建模创新

#### 2.1 自适应低通/高通滤波
- **代表**：`FreqFusion (TPAMI 2024)` → `FreqFusion(hr_channels, lr_channels)`
  - **ALPF**：上采样平滑，减少类内伪影
  - **AHPF**：增强边界细节
  - **Offset Generator**：局部相似性引导的特征重采样
- **缝合思路**：做图像融合 / 超分时替换简单双线性插值

#### 2.2 傅里叶域处理
- **代表**：`FourierUnit_modified (ICCV 2023)`、`FrequencyAttention (Arxiv 2025)`
- **创新点**：FFT 变换后在频域做增强，捕获周期模式
- **缝合思路**：遥感图像处理、医学图像增强

#### 2.3 小波变换（Wavelet）
- **代表**：`WTFD (Arxiv 2024)`（小波高低频分解）、`SFFNet`（空频融合）
- **创新点**：DWT 多尺度分解后分别处理高频/低频信息
- **缝合思路**：图像去噪、去雨、去雪

#### 2.4 DCT（离散余弦变换）
- **代表**：`DCTHighFrequencySpatialAttention`（二次创新组件）
- **创新点**：固定 DCT 频域掩码，抑制低频保留高频细节
- **缝合思路**：作为即插即用空间注意力替换 CBAM

---

### 3. 卷积形态演进

#### 3.1 深度可分离卷积（DWConv）
- **代表**：`DWConv (CVPR 2017)`、`PConv (CVPR 2023)`、`DSConv (CVPR 2023)`
- **创新点**：逐通道卷积 + 点卷积，大幅减少参数量
- **缝合思路**：轻量化模型 backbone 中替换标准卷积

#### 3.2 可变形卷积（DCN）
- **代表**：`DCNv2 (CVPR 2019)`、`DCNv4 (CVPR 2024)`
- **创新点**：可学习的偏移量让卷积核自适应变形
- **缝合思路**：目标检测、实例分割中替换常规卷积

#### 3.3 条带/条形卷积（Strip Conv）
- **代表**：`strip_pooling (CVPR 2020)`、`SSAx`、`SAFM`
- **创新点**：1×k 或 k×1 的非对称核捕获细长结构
- **缝合思路**：遥感影像、道路/河流提取

#### 3.4 多核并行卷积
- **代表**：`EMCAM (CVPR 2024)`（1+3+5 多尺度卷积）、`PSConv (AAAI 2025)`（风车形状）
- **创新点**：多个不同尺寸卷积核并行，捕获不同感受野
- **缝合思路**：替换 backbone 中的 3×3 卷积

---

### 4. 门控机制创新

#### 4.1 GLU 及其变体
- **代表**：`CGLU (CVPR 2024)` → `CGLU(in_features)`（卷积门控线性单元）
- **创新点**：在 GLU 的门控分支中加入 DWConv，增强局部建模
- **缝合思路**：替换 FFN 层，增强局部-全局信息混合

#### 4.2 SwiGLU
- **代表**：`SwiGLU`（MHDA/DiffTransformer 中的 FFN）
- **创新点**：Swish 激活函数替代 ReLU，提升非线性表达能力
- **缝合思路**：NLP/多模态模型的 FFN 替换

#### 4.3 通道门控
- **代表**：`SE (CVPR 2018)` → `SEAttention`、`SKAttention`
- **创新点**：Squeeze-and-Excitation 动态校准通道权重
- **缝合思路**：几乎所有视觉 backbone 的标配

#### 4.4 空间门控
- **代表**：`CBAM (ECCV 2018)`（通道+空间双门控）
- **缝合思路**：即插即用到任意 CNN 特征提取后

---

### 5. Mamba / 状态空间模型

#### 5.1 标准 Mamba
- **代表**：`CSMamba`（通道-空间 Mamba）、`PVMamba (Arxiv 2024)`
- **创新点**：选择性扫描（selective scan）实现 O(n) 长程建模
- **缝合思路**：序列建模、遥感语义分割、视频理解

#### 5.2 Mamba + 注意力混合
- **代表**：`MLLAttention`（Mamba-like Linear Attention）
- **创新点**：融合 LePE（局部增强）+ RoPE（旋转位置编码）+ 线性注意力
- **缝合思路**：替换 ViT 中的标准 self-attention

#### 5.3 KAN + Mamba
- **代表**：`MKLA (Arxiv 2024)` → `MKLAttention`
- **创新点**：用 KAN（B 样条基函数）替代 QK 线性投影
- **缝合思路**：实验性模块，学术创新价值高

---

### 6. 多尺度特征处理

#### 6.1 特征金字塔（FPN / ASPP / PPM）
- **代表**：`SAFM (ICCV 2023)`（自适应多尺度融合）、`PPA (Arxiv 2024)`
- **创新点**：不同下采样率分支，捕获多尺度语义

#### 6.2 频域+空域双域融合
- **代表**：`SFFNet (Arxiv 2024)` → `MDAF`（多尺度双表示对齐滤波器）
- **创新点**：小波分解后的多尺度空频特征对齐
- **缝合思路**：遥感影像语义分割

#### 6.3 局部-全局融合
- **代表**：`FreqFusion`（局部相似性引导采样）
- **缝合思路**：密集预测任务（分割、检测、深度估计）

---

### 7. 即插即用缝合范式

#### 7.1 串行缝合（Serial Suture）
```
输入 → [模块A] → [模块B] → [模块C] → 输出
```
- **示例**：`ScConv_EMA` → SRU(门控) → EMA(多尺度) → CRU(信道重构)
- **适用场景**：特征需要逐步精炼的任务

#### 7.2 并行缝合（Parallel Suture）
```
输入 → [分支A] ─┬─→ [融合] → 输出
            └──→ [分支B] ─┘
```
- **融合方式**：
  - 相加：`x = A(x) + B(x)`
  - 拼接：`x = Conv(Cat([A(x), B(x)]))`
  - 门控：`x = A(x) * sigmoid(B(x))`
- **示例**：`SEMAConv` → ScConv * EMA 门控相乘
- **适用场景**：需要多路径增强的任务

#### 7.3 自适应融合（Adaptive Fusion）
```
x1 = A(x), x2 = B(x)
w = sigmoid(Conv(Cat([x1, x2])))
output = x1 * w + x2 * (1 - w)
```
- **示例**：`GJFH` → 自适应权重动态平衡两分支

#### 7.4 残差连接 + 即插即用
- 几乎所有模块都支持：`output = module(input) + input`
- 即插即用的前提：输入输出 shape 完全一致

---

### 8. 微创新组件库（`二次创新/`）

用于在已有模块基础上做增量创新（增删改）：

| 组件 | 创新方向 | 用法 |
|------|----------|------|
| `两种特征降维方式` | Linear / Conv1x1 替换 Cat 后的维度 | 在 Cat 后加 ChannelReducer |
| `小波提取特征` | 在 H/V/对角方向上分解 | 替换 Spatial Attention |
| `傅里叶频域过滤` | 可学习频域权重调制 | 在 Transformer 中加 FFT 分支 |
| `DCT高频注意力` | 固定 DCT 掩码保留高频 | 替换空间注意力 |
| `基于余弦相似度加权` | 特征相似性引导重标定 | 特征融合后加权 |
| `门控掩码机制` | 信息量驱动的动态掩码 | 在残差前加 Gate |
| `通道分解与缩放` | 分解冗余通道 + 尺度变换 | 替换 heavy MLP |
| `欧拉公式特征` | 虚实双通道正交建模 | 替换部分 MLP 层 |
| `位置编码` | H/V 方向的正弦编码 | 加到特征图上 |

---

### 9. 任务特化模块

| 任务 | 推荐模块 | 核心机制 |
|------|----------|----------|
| **图像恢复** | Restormer, SSA, FreqFusion | 通道注意力 / 条带注意力 |
| **目标检测** | CGLU, DySample, MCAttention | 轻量化 + 小目标 |
| **语义分割** | CSMamba, SAFM, BiFormer | 多尺度 + 长程建模 |
| **超分辨率** | EfficientAttention, FreqFusion | 高效注意力 + 频域 |
| **小目标检测** | MCAttention, FCM, MKP | Monte Carlo 采样 |
| **遥感影像** | CSMamba, WTFD, SFFNet | Mamba + 小波 |
| **医学影像** | EMCAM, MCAttention | 多尺度 + 随机池化 |
| **时间序列** | ScaleGraphBlock, SSM | 图神经网络 + SSM |

---

### 10. 创新组合公式

```
新模块 = 主干机制 + 辅助增强 + 融合策略

主干机制选择：
  • Self-Attention（通用，Transformer 标准）
  • Mamba（长序列，O(n) 复杂度）
  • 卷积（局部建模，轻量）

辅助增强（可叠加）：
  • + RoPE / CoPE（位置编码）
  • + DWConv（局部增强）
  • + RMSNorm（稳定训练）
  • + GLU/SwiGLU（门控 FFN）

融合策略：
  • 串行：渐进精炼
  • 并行相加：多路径增强
  • 门控相乘：动态选择
  • 自适应加权：soft 平衡
```

---

### 11. 顶会论文创新点速查

| 年份 | 顶会 | 核心创新 | 模块名 |
|------|------|----------|--------|
| 2017 | CVPR | 深度可分离卷积 | DWConv |
| 2018 | CVPR | SE 注意力 | SE |
| 2018 | ECCV | CBAM 通道+空间双注意力 | CBAM |
| 2019 | CVPR | SK 注意力（多分支融合） | SKAttention |
| 2019 | CVPR | 可变形卷积 v2 | DCNv2 |
| 2020 | CVPR | Strip Pooling（条带池化） | strip_pooling |
| 2021 | CVPR | CoT / Coord / Triplet 注意力 | CoTAttention |
| 2022 | CVPR | Restormer（通道注意力 + GDFN） | Restormer |
| 2022 | CVPR | ACmix（卷积+注意力融合） | ACmix |
| 2022 | ICLR | ODConv（动态卷积） | ODConv |
| 2023 | CVPR | PConv / DSConv / ScConv | PConv, ScConv |
| 2023 | CVPR | BiFormer（双层路由注意力） | BiFormer |
| 2023 | ICCV | DySample / SAFM / LiteMLA | DySample |
| 2024 | CVPR | ASSA / CAA / FRFN | ASSA, CAA |
| 2024 | CVPR | CGLU（卷积门控） | CGLU |
| 2024 | ECCV | Agent-Attention（代理令牌） | AgentAttention |
| 2024 | TPAMI | FreqFusion（频域感知融合） | FreqFusion |
| 2024 | ArXiv | MHDA（差分注意力） | MHDA |
| 2024 | ArXiv | MLLA / MKLA（Mamba+KAN） | MLLAttention |
| 2025 | CVPR | Mona / GSA / DBlock / EBlock | Mona |
| 2025 | ECCV | SFHformer / DHSA | DHSA |
| 2026 | AAAI | StripeAttentionBlock | StripeAttentionBlock |

---

*本文件由 `scripts/build_index.py` 自动生成 last_updated: 2026-05-31*
