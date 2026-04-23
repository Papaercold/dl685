import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data_utils import build_dataset_loaders
from hyperinr import CoordNetConfig, HyperNetwork, total_param_count


class LinearClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 10) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def extract_features(
    model: Optional[HyperNetwork],
    loader,
    device: torch.device,
    feature_type: str,
    max_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if model is not None:
        model.eval()
    feats: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    seen = 0

    with torch.no_grad():
        for images, y in loader:
            images = images.to(device, non_blocking=True)
            if feature_type == "pixels":
                feature = images.view(images.size(0), -1).cpu()
            else:
                assert model is not None
                out = model(images)
                feature = out["latent"] if feature_type == "latent" else out["theta"]
                feature = feature.cpu()
            feats.append(feature)
            labels.append(y)
            seen += images.size(0)
            if max_samples > 0 and seen >= max_samples:
                break

    x = torch.cat(feats, dim=0)
    y = torch.cat(labels, dim=0)
    if max_samples > 0:
        x = x[:max_samples]
        y = y[:max_samples]
    return x, y


def train_classifier(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_loader, test_loader = build_dataset_loaders(
        dataset="cifar10",
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=32,
    )

    if args.feature_type == "pixels":
        hyper = None
    else:
        ckpt = torch.load(args.recon_ckpt, map_location="cpu")
        cfg = CoordNetConfig(**ckpt["coord_cfg"])
        hyper = HyperNetwork(total_param_count(cfg), latent_dim=ckpt["args"]["latent_dim"], coord_cfg=cfg).to(device)
        hyper.load_state_dict(ckpt["model"])

    train_x, train_y = extract_features(hyper, train_loader, device, args.feature_type, args.max_train_samples)
    test_x, test_y = extract_features(hyper, test_loader, device, args.feature_type, args.max_test_samples)

    train_ds = TensorDataset(train_x, train_y)
    test_ds = TensorDataset(test_x, test_y)
    tr_loader = DataLoader(train_ds, batch_size=args.cls_batch_size, shuffle=True)
    te_loader = DataLoader(test_ds, batch_size=args.cls_batch_size, shuffle=False)

    classifier = LinearClassifier(in_dim=train_x.size(1), num_classes=10).to(device)
    opt = torch.optim.Adam(classifier.parameters(), lr=args.cls_lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, args.cls_epochs + 1):
        classifier.train()
        for x, y in tr_loader:
            x = x.to(device)
            y = y.to(device)
            logits = classifier(x)
            loss = ce(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in te_loader:
                x = x.to(device)
                y = y.to(device)
                pred = classifier(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()
        acc = 100.0 * correct / max(total, 1)
        best_acc = max(best_acc, acc)
        print(f"epoch={epoch:03d} val_acc={acc:.2f}% best_acc={best_acc:.2f}%")

    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "classifier": classifier.state_dict(),
            "feature_type": args.feature_type,
            "in_dim": train_x.size(1),
            "best_acc": best_acc,
            "recon_ckpt": getattr(args, "recon_ckpt", None),
        },
        out,
    )
    print(f"saved classifier: {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Downstream CIFAR-10 classification from hypernetwork features")
    p.add_argument("--recon-ckpt", type=str, default=None, help="Required for latent/theta feature types.")
    p.add_argument("--feature-type", type=str, default="latent", choices=["latent", "theta", "pixels"])
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--out-path", type=str, default="./runs/cifar10_downstream.pt")

    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-train-samples", type=int, default=10000)
    p.add_argument("--max-test-samples", type=int, default=2000)

    p.add_argument("--cls-epochs", type=int, default=20)
    p.add_argument("--cls-batch-size", type=int, default=256)
    p.add_argument("--cls-lr", type=float, default=1e-3)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train_classifier(parse_args())
