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
    # Encoder: 3→64→128→256 channels, 3 stride-2 convs, spatial pool to 4×4.
    # For a 32px input  : feature map 4×4 after striding (pool is identity).
    # For a 128px input : feature map 16×16 after striding, pooled to 4×4.
    # Either way the encoder output is always 256×4×4 = 4096-dim.
    _ENC_CH = 256
    _ENC_SP = 4

    def __init__(
        self,
        total_params: int,
        latent_dim: int = 256,
        coord_cfg: Optional[CoordNetConfig] = None,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        enc_out = self._ENC_CH * self._ENC_SP * self._ENC_SP  # 4096

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, self._ENC_CH, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self._ENC_CH),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((self._ENC_SP, self._ENC_SP)),
            nn.Flatten(),
        )
        # LayerNorm after projection keeps z in a predictable range, which
        # makes SIREN theta_head initialization exact (see below).
        self.latent_head = nn.Sequential(
            nn.Linear(enc_out, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(inplace=True),
        )
        self.theta_head = nn.Linear(latent_dim, total_params)

        # Calibrated theta_head init for ALL architectures.
        # ReLU in latent_head kills ~half the neurons → E[z²] ≈ 0.5.
        # Var(theta_i) = latent_dim * 0.5 * W_std²  →  theta_std ≈ sqrt(latent_dim/2)*W_std.
        # W_std = sqrt(2/latent_dim)  →  theta_std ≈ 1.
        # Per-layer scaling in unpack_params then maps theta to the correct
        # weight range for each architecture (SIREN range or Kaiming range).
        nn.init.normal_(self.theta_head.weight, std=math.sqrt(2.0 / latent_dim))
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

        # Per-layer weight scaling so every architecture starts in a healthy range.
        # Without this, ReLU/Fourier hidden weights are 3× too large → activations
        # explode across 4 layers → float16 overflow → NaN loss by epoch 3.
        if idx < n_layers - 1:
            if cfg.arch == "siren":
                # Sitzmann et al. 2020: uniform in [-1/n, 1/n] for first layer,
                # [-sqrt(6/n)/ω, sqrt(6/n)/ω] for hidden layers.
                scale = 1.0 / in_dim if idx == 0 else math.sqrt(6.0 / in_dim) / cfg.hidden_omega
            else:
                # Kaiming / He init for ReLU and Fourier MLPs.
                scale = math.sqrt(2.0 / in_dim)
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
