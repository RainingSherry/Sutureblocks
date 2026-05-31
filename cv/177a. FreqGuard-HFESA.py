import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class FreqGuardHFESA(nn.Module):
    """Frequency-Guarded HFESA.

    This modified module keeps HFESA's same [B, C, H, W] input/output contract.
    It adds a frequency-energy gate and a directional high-frequency residual
    guard, so the low-resolution attention branch is less likely to wash out
    texture, edges, rain streaks, blur boundaries, or UHD fine structures.
    """

    def __init__(self, dim, num_heads=8, bias=False, window_size=8, reduction=4):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.dwconv_w = nn.Conv2d(dim, dim, kernel_size=(1, 11), padding=(0, 5), groups=dim)
        self.dwconv_h = nn.Conv2d(dim, dim, kernel_size=(11, 1), padding=(5, 0), groups=dim)
        self.dwconv_hw = nn.Conv2d(dim * 2, dim * 2, 3, padding=1, groups=dim * 2)
        self.conv1 = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        self.sr = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=bias)

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.conv4 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias),
        )

        hidden = max(dim // reduction, 8)
        self.freq_gate = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, dim, 1, bias=True),
            nn.Sigmoid(),
        )
        self.direction_guard = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(1, 7), padding=(0, 3), groups=dim, bias=bias),
            nn.Conv2d(dim, dim, kernel_size=(7, 1), padding=(3, 0), groups=dim, bias=bias),
            nn.Conv2d(dim, dim, 1, bias=bias),
        )
        self.local_blur = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.guard_scale = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

    def _frequency_gate(self, x):
        # 用 rFFT 的幅值谱估计通道高频能量，再生成全局频谱门控。
        spectrum = torch.fft.rfft2(x.float(), norm="ortho")
        energy = torch.log1p(torch.abs(spectrum)).mean(dim=(-2, -1), keepdim=True)
        return self.freq_gate(energy.to(dtype=x.dtype))

    def forward(self, x):
        b, c, h, w = x.shape

        high = self.dwconv_hw(torch.cat([self.dwconv_w(x), self.dwconv_h(x)], dim=1))
        high = self.conv1(high)

        x_down = self.sr(x)
        qkv = self.qkv_dwconv(self.qkv(x_down))
        q, k, v_down = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        freq_gate = self._frequency_gate(x)
        spatial_hf = x - self.local_blur(x)
        guarded_hf = self.direction_guard(spatial_hf) * freq_gate

        v = self.conv2(self.max(v_down)) + self.conv3(self.avg(v_down))
        v = v * (self.conv4(x) + guarded_hf)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        low = attn @ v
        low = rearrange(low, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)

        fused = self.project_out(torch.cat([high, low], dim=1))
        return fused + self.guard_scale * guarded_hf


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 模拟 UHD 图像恢复中间层特征输入 [B, C, H, W]
    input_tensor = torch.randn(2, 64, 32, 32).to(device)

    # 直接实例化真正的核心创新模块：FreqGuardHFESA
    model = FreqGuardHFESA(dim=64, num_heads=8, bias=False, window_size=8).to(device)

    print(model)

    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)
    print("output_tensor_shape :", output_tensor.shape)

    print("\n哔哩哔哩/微信公众号: CV缝合救星, 独家整理! \n")
