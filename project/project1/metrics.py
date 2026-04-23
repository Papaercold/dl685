import math
from typing import Callable, Optional

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target).item()
    return 10.0 * math.log10(1.0 / max(mse, 1e-10))


def _gaussian_kernel(window_size: int, sigma: float, channels: int, device: torch.device) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g.outer(g)                             # [W, W]
    return kernel.expand(channels, 1, window_size, window_size)  # [C, 1, W, W]


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    c1: float = 0.01 ** 2,
    c2: float = 0.03 ** 2,
) -> float:
    """Window-based SSIM (Wang et al. 2004). Inputs expected in [0, 1]."""
    channels = pred.size(1)
    pad = window_size // 2
    kernel = _gaussian_kernel(window_size, sigma, channels, pred.device)

    mu_x = F.conv2d(pred, kernel, padding=pad, groups=channels)
    mu_y = F.conv2d(target, kernel, padding=pad, groups=channels)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, kernel, padding=pad, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(target * target, kernel, padding=pad, groups=channels) - mu_y2
    sigma_xy = F.conv2d(pred * target, kernel, padding=pad, groups=channels) - mu_xy

    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return (numerator / denominator).mean().item()


def build_lpips_metric(device: torch.device) -> Optional[Callable[[torch.Tensor, torch.Tensor], float]]:
    try:
        import lpips  # type: ignore
    except Exception:
        return None

    metric = lpips.LPIPS(net="alex").to(device)
    metric.eval()

    def _call(pred: torch.Tensor, target: torch.Tensor) -> float:
        with torch.no_grad():
            x = pred * 2.0 - 1.0
            y = target * 2.0 - 1.0
            score = metric(x, y).mean().item()
        return score

    return _call
