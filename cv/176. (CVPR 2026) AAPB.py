import torch
import torch.nn as nn
from typing import Optional


class AdaptiveAuxiliaryPromptBlender(nn.Module):
    """AAPB-style score-space prompt blender.

    The module consumes unconditional, target-conditioned, and auxiliary-anchor
    scores from a diffusion model and returns a blended guided score with the
    same shape. It is training-free and can sit inside a denoising loop.
    """

    def __init__(self, guidance_scale: float = 7.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.guidance_scale = guidance_scale
        self.eps = eps

    def adaptive_gamma(self, uncond_score: torch.Tensor, target_score: torch.Tensor, anchor_score: torch.Tensor) -> torch.Tensor:
        direction = anchor_score - target_score
        residual = target_score - uncond_score
        numerator = -(residual * direction).flatten(1).sum(dim=1)
        denominator = direction.square().flatten(1).sum(dim=1).clamp_min(self.eps)
        gamma = numerator / denominator
        return gamma.clamp(0.0, 1.0).view(-1, 1, 1, 1)

    def forward(
        self,
        uncond_score: torch.Tensor,
        target_score: torch.Tensor,
        anchor_score: torch.Tensor,
        guidance_scale: Optional[float] = None,
    ) -> torch.Tensor:
        scale = self.guidance_scale if guidance_scale is None else guidance_scale
        gamma = self.adaptive_gamma(uncond_score, target_score, anchor_score)
        conditional = (1.0 - gamma) * target_score + gamma * anchor_score
        return uncond_score + scale * (conditional - uncond_score)


def main() -> None:
    torch.manual_seed(11)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 模拟扩散模型 score/latent 输入 [B, C, H, W]
    input_tensor = torch.randn(1, 4, 32, 32).to(device)
    target_tensor = input_tensor + 0.2 * torch.randn_like(input_tensor)
    anchor_tensor = input_tensor + 0.2 * torch.randn_like(input_tensor)

    # 直接实例化真正的核心创新模块：AAPB
    model = AdaptiveAuxiliaryPromptBlender(guidance_scale=6.5).to(device)

    print(model)

    output_tensor = model(input_tensor, target_tensor, anchor_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)
    print("output_tensor_shape :", output_tensor.shape)
    print("\n哔哩哔哩/微信公众号: CV缝合救星, 独家整理! \n")


if __name__ == "__main__":
    main()
