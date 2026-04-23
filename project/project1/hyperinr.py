import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CoordNetConfig:
    arch: str = "siren"
    hidden_dim: int = 64
    depth: int = 3
    out_dim: int = 3
    first_omega: float = 30.0
    hidden_omega: float = 30.0
    fourier_features: int = 16
    fourier_sigma: float = 10.0

    @property
    def input_dim(self) -> int:
        if self.arch == "fourier":
            return self.fourier_features * 2
        return 2

    @property
    def layer_sizes(self) -> List[int]:
        dims = [self.input_dim]
        for _ in range(self.depth):
            dims.append(self.hidden_dim)
        dims.append(self.out_dim)
        return dims


class HyperNetwork(nn.Module):
    def __init__(
        self,
        total_params: int,
        latent_dim: int = 256,
        coord_cfg: Optional[CoordNetConfig] = None,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.latent_head = nn.Sequential(
            nn.Linear(128, latent_dim),
            nn.ReLU(inplace=True),
        )
        self.theta_head = nn.Linear(latent_dim, total_params)

        # SIREN: re-initialize theta_head so its output has std ≈ 1, which after
        # per-layer scaling in unpack_params lands weights in the correct SIREN range.
        # Default Xavier gives theta std ≈ 0.03–0.07, making hidden weights 30× too
        # small and collapsing SIREN hidden layers into the linear regime of sin().
        if coord_cfg is not None and coord_cfg.arch == "siren":
            nn.init.normal_(self.theta_head.weight, std=1.15)
            nn.init.zeros_(self.theta_head.bias)

        # Fixed random Fourier frequency matrix (Tancik et al. 2020).
        # Stored as a buffer so it moves with the model and is saved in checkpoints.
        if coord_cfg is not None and coord_cfg.arch == "fourier":
            B = torch.randn(2, coord_cfg.fourier_features) * coord_cfg.fourier_sigma
            self.register_buffer("fourier_B", B)
        else:
            self.fourier_B: Optional[torch.Tensor] = None

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        base = self.encoder(images)
        return self.latent_head(base)

    def forward(self, images: torch.Tensor) -> Dict[str, object]:
        z = self.encode(images)
        theta = self.theta_head(z)
        out: Dict[str, object] = {"latent": z, "theta": theta}
        if self.fourier_B is not None:
            out["fourier_B"] = self.fourier_B
        return out


def total_param_count(cfg: CoordNetConfig) -> int:
    total = 0
    sizes = cfg.layer_sizes
    for in_dim, out_dim in zip(sizes[:-1], sizes[1:]):
        total += out_dim * in_dim
        total += out_dim
    return total


def unpack_params(
    theta: torch.Tensor, cfg: CoordNetConfig
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    batch = theta.size(0)
    cursor = 0
    params: List[Tuple[torch.Tensor, torch.Tensor]] = []
    sizes = cfg.layer_sizes
    n_layers = len(sizes) - 1

    for idx, (in_dim, out_dim) in enumerate(zip(sizes[:-1], sizes[1:])):
        w_size = out_dim * in_dim
        b_size = out_dim

        w = theta[:, cursor : cursor + w_size].view(batch, out_dim, in_dim)
        cursor += w_size
        b = theta[:, cursor : cursor + b_size].view(batch, out_dim)
        cursor += b_size

        # SIREN weight scaling (Sitzmann et al. 2020).
        # Multiply by the target initialization scale so weights start in the
        # right range; no tanh — clamping every forward pass prevents the
        # network from learning high-frequency content (worse LPIPS).
        if cfg.arch == "siren" and idx < n_layers - 1:
            if idx == 0:
                scale = 1.0 / in_dim
            else:
                scale = math.sqrt(6.0 / in_dim) / cfg.hidden_omega
            w = w * scale
            b = b * scale

        params.append((w, b))

    if cursor != theta.size(1):
        raise ValueError("Failed to unpack generated coordinate-network parameters.")
    return params


def batched_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    return torch.bmm(x, weight.transpose(1, 2)) + bias.unsqueeze(1)


def fourier_encode(coords: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Random Fourier feature encoding (Tancik et al. 2020).

    coords: [batch, n_points, 2]  in [-1, 1]
    B:      [2, n_features]       fixed random frequency matrix
    returns [batch, n_points, 2*n_features]
    """
    proj = torch.matmul(coords, B) * (2.0 * math.pi)
    return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


def coord_forward(
    coords: torch.Tensor,
    params: List[Tuple[torch.Tensor, torch.Tensor]],
    cfg: CoordNetConfig,
    fourier_B: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if cfg.arch == "fourier":
        if fourier_B is None:
            raise ValueError("fourier_B must be provided for fourier arch.")
        x = fourier_encode(coords, fourier_B)
    else:
        x = coords

    for idx, (w, bias) in enumerate(params):
        x = batched_linear(x, w, bias)
        is_last = idx == len(params) - 1
        if not is_last:
            if cfg.arch == "siren":
                omega = cfg.first_omega if idx == 0 else cfg.hidden_omega
                x = torch.sin(omega * x)
            else:
                x = F.relu(x, inplace=False)

    return torch.sigmoid(x)
