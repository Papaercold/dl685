from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class _CelebAFolder(Dataset):
    """Read CelebA images directly from img_align_celeba/, no Google Drive needed.

    Uses the official split boundaries from list_eval_partition.txt:
      train 000001–162770, valid 162771–182637, test 182638–202599.
    Files are named sequentially so sorted order == split order.
    """

    _SPLIT_END = {"train": 162770, "valid": 182637, "test": 202599}

    def __init__(self, root: str, split: str, transform=None):
        self.transform = transform
        img_dir = Path(root) / "celeba" / "img_align_celeba"
        if not img_dir.exists():
            raise FileNotFoundError(
                f"CelebA images not found at {img_dir}\n"
                "Run the 'CelebA Setup' cell in Section 4 of the notebook first."
            )
        all_paths = sorted(img_dir.glob("*.jpg"))
        if len(all_paths) == 0:
            raise FileNotFoundError(f"No .jpg files in {img_dir}")

        # boundaries (1-based → 0-based indices)
        prev = {"train": 0, "valid": 162770, "test": 182637}
        end = self._SPLIT_END
        self.paths = all_paths[prev[split] : end[split]]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0  # dummy label — reconstruction only


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
        train_set = _CelebAFolder(root=data_root, split="train", transform=tf)
        test_set = _CelebAFolder(root=data_root, split="valid", transform=tf)
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
