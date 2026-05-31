import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MatchedCrispEdgeModule(nn.Module):
    """Lightweight MatchED-style edge head.

    The CVPR 2026 paper describes a five-block lightweight CNN followed by a
    sigmoid edge map. The matching formulation is a training loss; this module
    keeps the deployable prediction head compact and plug-and-play.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 32, blocks: int = 5) -> None:
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.refine = nn.Sequential(*[ConvNormAct(hidden_channels) for _ in range(blocks)])
        self.exit = nn.Conv2d(hidden_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.entry(x)
        feat = self.refine(feat)
        return torch.sigmoid(self.exit(feat))


def soft_chamfer_edge_loss(pred: torch.Tensor, target: torch.Tensor, radius: int = 3) -> torch.Tensor:
    """A differentiable local proxy for MatchED's one-to-one spatial matching loss."""
    pred = pred.clamp(1e-4, 1.0 - 1e-4)
    target = target.float()

    # Local target support rewards predictions that land near a true edge pixel.
    target_support = F.max_pool2d(target, kernel_size=2 * radius + 1, stride=1, padding=radius)
    pred_support = F.max_pool2d(pred, kernel_size=2 * radius + 1, stride=1, padding=radius)

    precision_term = -(target_support * torch.log(pred) + (1.0 - target_support) * torch.log(1.0 - pred)).mean()
    recall_term = (target * (1.0 - pred_support)).sum() / target.sum().clamp_min(1.0)
    crisp_term = F.avg_pool2d(pred, 3, stride=1, padding=1).sub(pred).abs().mean()
    return precision_term + recall_term + 0.1 * crisp_term


def main() -> None:
    torch.manual_seed(7)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 模拟边缘检测输入 [B, 3, 256, 256]
    input_tensor = torch.randn(1, 3, 256, 256).to(device)

    # 直接实例化真正的核心创新模块：MatchED
    model = MatchedCrispEdgeModule(in_channels=3).to(device)

    print(model)

    output_tensor = model(input_tensor)
    target = (torch.rand_like(output_tensor) > 0.97).float()
    loss = soft_chamfer_edge_loss(output_tensor, target)
    loss.backward()

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)
    print("output_tensor_shape :", output_tensor.shape)
    print("\n哔哩哔哩/微信公众号: CV缝合救星, 独家整理! \n")


if __name__ == "__main__":
    main()
