import torch
import torch.nn as nn
import torch.nn.functional as F

# 辅助函数：图像转窗口
def img2windows(img, H_sp, W_sp):
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    return img_perm

# 辅助函数：窗口转图像
def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))
    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img

class DynamicPosBias(nn.Module):
    """ 动态位置偏置模块 (参考 Crossformer) """
    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(nn.LayerNorm(self.pos_dim), nn.ReLU(inplace=True), nn.Linear(self.pos_dim, self.pos_dim))
        self.pos2 = nn.Sequential(nn.LayerNorm(self.pos_dim), nn.ReLU(inplace=True), nn.Linear(self.pos_dim, self.pos_dim))
        self.pos3 = nn.Sequential(nn.LayerNorm(self.pos_dim), nn.ReLU(inplace=True), nn.Linear(self.pos_dim, self.num_heads))
        
    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases)
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos

class FractalAttention(nn.Module):
    """
    魔改创新点 1: 分形结构与多粒度注意力融合
    """
    def __init__(self, dim, idx, split_size=[8,8], num_heads=6, attn_drop=0., position_bias=True):
        super().__init__()
        self.dim = dim
        self.split_size = split_size
        self.num_heads = num_heads
        self.idx = idx
        self.position_bias = position_bias

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        if idx == 0:
            self.H_sp, self.W_sp = self.split_size[0], self.split_size[1]
        else:
            self.H_sp, self.W_sp = self.split_size[1], self.split_size[0]

        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)
            # 生成相对位置坐标母集
            position_bias_h = torch.arange(1 - self.H_sp, self.H_sp)
            position_bias_w = torch.arange(1 - self.W_sp, self.W_sp)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w], indexing='ij'))
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()
            self.register_buffer('rpe_biases', biases)

            # 获取窗口内每对 token 的相对位置索引
            coords_h = torch.arange(self.H_sp)
            coords_w = torch.arange(self.W_sp)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.H_sp - 1
            relative_coords[:, :, 1] += self.W_sp - 1
            relative_coords[:, :, 0] *= 2 * self.W_sp - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer('relative_position_index', relative_position_index)

        # 魔改点: 分形空洞卷积提取多尺度曲线特征
        self.fractal_conv1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.fractal_conv2 = nn.Conv2d(dim, dim, 3, padding=2, dilation=2, groups=dim, bias=False)
        
        # 魔改点: 动态频率选择权重 (取代固定的 beta)
        self.dynamic_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, 1, 1),
            nn.Sigmoid()
        )
        
        self.alpha = nn.Parameter(torch.zeros(1))
        self.attn_drop = nn.Dropout(attn_drop)

    def im2win(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, qkv, H, W, mask=None):
        q, k, v_raw = qkv[0], qkv[1], qkv[2]
        B, L, C = q.shape

        # 魔改点：提取分形引导特征
        x2d = q.transpose(1, 2).view(B, C, H, W)
        f_feat = self.fractal_conv1(x2d) + self.fractal_conv2(x2d)
        
        # 动态计算混合门控权重
        gate = self.dynamic_gate(f_feat).view(B, 1, 1, 1) # [B, 1, 1, 1]

        q = self.im2win(q, H, W)
        k = self.im2win(k, H, W)
        v = self.im2win(v_raw, H, W)

        # 基础注意力得分
        attn_logits = (q @ k.transpose(-2, -1)) * self.scale

        if self.position_bias:
            pos = self.pos(self.rpe_biases)
            relative_position_bias = pos[self.relative_position_index.view(-1)].view(
                self.H_sp * self.W_sp, self.H_sp * self.W_sp, -1).permute(2, 0, 1).contiguous()
            attn_logits = attn_logits + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn_logits = attn_logits.view(B, nW, self.num_heads, -1, -1) + mask.unsqueeze(1).unsqueeze(0)
            attn_logits = attn_logits.view(-1, self.num_heads, attn_logits.shape[-2], attn_logits.shape[-1])

        # 分形引导调制
        fractal_guide = img2windows(f_feat.mean(dim=1, keepdim=True), self.H_sp, self.W_sp)
        fractal_guide = fractal_guide.view(-1, 1, self.H_sp * self.W_sp, 1)
        modulated_logits = attn_logits * (1 + self.alpha * fractal_guide)

        # 动态混合标准注意力和引导注意力
        attn_std = F.softmax(attn_logits, dim=-1)
        attn_mgf = F.softmax(modulated_logits, dim=-1)
        
        # 使用动态 gate 进行加权
        attn = gate * attn_std + (1 - gate) * attn_mgf
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)
        x = windows2img(x, self.H_sp, self.W_sp, H, W)
        
        return x

class MGF_Attn(nn.Module):
    """
    Multi-Granularity Fractal Attention (MGF-Attn)
    CVPR 2026 风格魔改模块 - 针对复杂曲线结构的增强设计
    """
    def __init__(self, dim, num_heads, split_size=[4,4], shift_size=[2,2], qkv_bias=False, reso=64, rs_id=0, idx=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.split_size = split_size
        self.shift_size = shift_size
        self.idx = idx
        self.rs_id = rs_id
        self.patches_resolution = reso
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.)

        # 双分支注意力：通过不同的 split_size 捕捉多粒度特征
        self.attns = nn.ModuleList([
            FractalAttention(
                dim // 2, idx=i, split_size=split_size, 
                num_heads=num_heads // 2, position_bias=True)
            for i in range(2)
        ])

        # 魔改点: 局部结构增强混合 (LSE Mix)
        self.lse_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

    def forward(self, x, H, W):
        B, L, C = x.shape
        # 生成 QKV
        qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3) # [3, B, HW, C]
        v_orig = qkv[2].transpose(-2, -1).contiguous().view(B, C, H, W)

        # 处理 Padding 以适应窗口切分
        max_sp = max(self.split_size)
        pad_r = (max_sp - W % max_sp) % max_sp
        pad_b = (max_sp - H % max_sp) % max_sp
        
        qkv_pad = F.pad(qkv.reshape(3*B, C, H, W), (0, pad_r, 0, pad_b))
        _H, _W = H + pad_b, W + pad_r
        qkv_pad = qkv_pad.view(3, B, C, _H, _W).permute(0, 1, 3, 4, 2) # [3, B, H, W, C]

        # 分支并行处理 (特征分裂)
        qkv_0 = qkv_pad[:, :, :, :, :C//2].reshape(3, B, -1, C//2)
        qkv_1 = qkv_pad[:, :, :, :, C//2:].reshape(3, B, -1, C//2)

        x1 = self.attns[0](qkv_0, _H, _W)[:, :H, :W, :].reshape(B, L, C//2)
        x2 = self.attns[1](qkv_1, _H, _W)[:, :H, :W, :].reshape(B, L, C//2)

        # 特征合并
        attened_x = torch.cat([x1, x2], dim=2)

        # 局部结构增强路径
        lse_feat = self.lse_conv(v_orig).permute(0, 2, 3, 1).contiguous().view(B, L, C)

        # 残差混合与投影
        x = attened_x + lse_feat
        x = self.proj(x)
        return self.proj_drop(x)

# 测试代码
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 输入维度: [Batch, tokens, channels]
    input_tensor = torch.randn(1, 1024, 128).to(device) 
    
    # 实例化 MGF-Attn
    model = MGF_Attn(dim=128, num_heads=8, split_size=[8, 8]).to(device)
    
    # 执行前向传播，假设分辨率是 32x32
    output = model(input_tensor, 32, 32)

    print(model)
    print(f"输入维度: {input_tensor.shape}")
    print(f"输出维度: {output.shape}")
    print("\n模块名称: MGF-Attn (Multi-Granularity Fractal Attention)")
    print("CV缝合救星原创: 分形空洞卷积引导 + 动态门控频率解耦")