from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_dataset_loaders(
    dataset: str,
    data_root: str,
    batch_size: int,
    num_workers: int,
    image_size: int,
) -> Tuple[DataLoader, DataLoader]:
    if dataset == "cifar10":
        tf = transforms.ToTensor()
        train_set = datasets.CIFAR10(root=data_root, train=True, download=True, transform=tf)
        test_set = datasets.CIFAR10(root=data_root, train=False, download=True, transform=tf)
    elif dataset == "celeba":
        tf = transforms.Compose(
            [
                transforms.CenterCrop(178),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )
        train_set = datasets.CelebA(root=data_root, split="train", download=True, transform=tf)
        test_set = datasets.CelebA(root=data_root, split="valid", download=True, transform=tf)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


def make_grid(size: int, device: torch.device) -> torch.Tensor:
    xs = torch.linspace(-1.0, 1.0, size, device=device)
    ys = torch.linspace(-1.0, 1.0, size, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=-1).reshape(1, size * size, 2)


def sample_random_coords(batch: int, n_samples: int, device: torch.device) -> torch.Tensor:
    return torch.rand(batch, n_samples, 2, device=device) * 2.0 - 1.0


def sample_image_at_coords(images: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    bsz, n_coords, _ = coords.shape
    grid = coords.view(bsz, n_coords, 1, 2)
    sampled = F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled.squeeze(-1).permute(0, 2, 1)


def downsample_images(images: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)
