import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseDirectionalBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.horizontal = nn.Conv2d(channels, channels, (1, 5), padding=(0, 2), groups=channels, bias=False)
        self.vertical = nn.Conv2d(channels, channels, (5, 1), padding=(2, 0), groups=channels, bias=False)
        self.mix = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mix(torch.cat([self.horizontal(x), self.vertical(x)], dim=1))


class FGTMMatchED(nn.Module):
    """Frequency-Guided Topology MatchED.

    Magic change: keep MatchED's input/output contract, then add a high-frequency
    confidence branch and a topology gate so thin edges are amplified only where
    local structure is coherent.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 32) -> None:
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.spatial_path = nn.Sequential(
            DepthwiseDirectionalBlock(hidden_channels),
            DepthwiseDirectionalBlock(hidden_channels),
        )
        self.freq_gate = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.Sigmoid(),
        )
        self.topology_gate = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.Sigmoid(),
        )
        self.exit = nn.Conv2d(hidden_channels, 1, 1)

    def _high_frequency(self, x: torch.Tensor) -> torch.Tensor:
        low = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        return x - low

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.entry(x)
        spatial = self.spatial_path(feat)
        high = self._high_frequency(feat)
        gate = self.freq_gate(high.abs())
        topo = self.topology_gate(torch.cat([spatial, high], dim=1))
        fused = feat + spatial * topo + high * gate
        return torch.sigmoid(self.exit(fused))


def topology_balanced_loss(pred: torch.Tensor, target: torch.Tensor, radius: int = 3) -> torch.Tensor:
    pred = pred.clamp(1e-4, 1.0 - 1e-4)
    target = target.float()
    target_support = F.max_pool2d(target, 2 * radius + 1, stride=1, padding=radius)
    bce = F.binary_cross_entropy(pred, target_support)
    local_mean = F.avg_pool2d(pred, 3, stride=1, padding=1)
    thinness = (pred * local_mean).mean()
    return bce + 0.05 * thinness


def main() -> None:
    torch.manual_seed(7)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 模拟边缘检测输入 [B, 3, 256, 256]
    input_tensor = torch.randn(1, 3, 256, 256).to(device)

    # 直接实例化真正的核心创新模块：FGTM-MatchED
    model = FGTMMatchED(in_channels=3).to(device)

    print(model)

    output_tensor = model(input_tensor)
    target = (torch.rand_like(output_tensor) > 0.97).float()
    loss = topology_balanced_loss(output_tensor, target)
    loss.backward()

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)
    print("output_tensor_shape :", output_tensor.shape)
    print("\n哔哩哔哩/微信公众号: CV缝合救星, 独家整理! \n")


if __name__ == "__main__":
    main()
