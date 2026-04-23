import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from data_utils import (
    build_dataset_loaders,
    downsample_images,
    make_grid,
    sample_image_at_coords,
    sample_random_coords,
)
from hyperinr import CoordNetConfig, HyperNetwork, coord_forward, total_param_count, unpack_params
from metrics import build_lpips_metric, psnr, ssim


def reconstruct_full(
    images: torch.Tensor,
    model: HyperNetwork,
    cfg: CoordNetConfig,
    output_size: int,
) -> torch.Tensor:
    bsz = images.size(0)
    device = images.device
    full_coords = make_grid(output_size, device).expand(bsz, -1, -1)

    out = model(images)
    params = unpack_params(out["theta"], cfg)
    fourier_B = out.get("fourier_B")
    recon_flat = coord_forward(full_coords, params, cfg, fourier_B=fourier_B)
    return recon_flat.view(bsz, output_size, output_size, 3).permute(0, 3, 1, 2)


def evaluate(
    model: HyperNetwork,
    cfg: CoordNetConfig,
    loader,
    device: torch.device,
    image_size: int,
    eval_batches: int,
    lpips_fn,
    superres_size: Optional[int],
    superres_lowres: Optional[int],
) -> Dict[str, float]:
    model.eval()
    stats: Dict[str, List[float]] = {"psnr": [], "ssim": []}
    if lpips_fn is not None:
        stats["lpips"] = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= eval_batches:
                break
            images = batch[0].to(device, non_blocking=True)

            if superres_size is None:
                recon = reconstruct_full(images, model, cfg, image_size)
                target = images
            else:
                if superres_lowres is None:
                    raise ValueError("superres_lowres must be set when superres evaluation is enabled.")
                low = downsample_images(images, superres_lowres)
                recon = reconstruct_full(low, model, cfg, superres_size)
                target = F.interpolate(images, size=(superres_size, superres_size), mode="bilinear", align_corners=False)

            stats["psnr"].append(psnr(recon, target))
            stats["ssim"].append(ssim(recon, target))
            if lpips_fn is not None:
                stats["lpips"].append(lpips_fn(recon, target))

    return {k: float(sum(v) / max(len(v), 1)) for k, v in stats.items()}


def train(args: argparse.Namespace) -> None:
    if args.superres_eval_size > 0 and args.train_lowres <= 0:
        raise ValueError("--superres-eval-size requires --train-lowres > 0.")
    if args.train_lowres > 0 and args.train_lowres >= args.image_size:
        raise ValueError("--train-lowres should be smaller than --image-size.")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_loader, test_loader = build_dataset_loaders(
        dataset=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    coord_cfg = CoordNetConfig(
        arch=args.arch,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        first_omega=args.first_omega,
        hidden_omega=args.hidden_omega,
        fourier_features=args.fourier_features,
        fourier_sigma=args.fourier_sigma,
    )
    total_params = total_param_count(coord_cfg)
    model = HyperNetwork(total_params=total_params, latent_dim=args.latent_dim, coord_cfg=coord_cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_superres{args.train_lowres}to{args.superres_eval_size}" if args.train_lowres > 0 else ""
    ckpt_path = out_dir / f"{args.dataset}_{args.arch}{suffix}_recon.pt"
    log_path = out_dir / f"{args.dataset}_{args.arch}{suffix}_recon_metrics.jsonl"

    lpips_fn = build_lpips_metric(device)
    if lpips_fn is None:
        print("[warn] lpips package not found; LPIPS metric will be skipped.")

    print(f"device={device} dataset={args.dataset} arch={args.arch} total_generated_params={total_params}")

    best_psnr = -1e9
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        start = time.time()

        for batch in train_loader:
            images = batch[0].to(device, non_blocking=True)
            bsz = images.size(0)

            input_images = images
            if args.train_lowres > 0:
                input_images = downsample_images(images, args.train_lowres)

            coords = sample_random_coords(bsz, args.samples_per_image, device)
            targets = sample_image_at_coords(input_images, coords)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                out = model(input_images)
                params = unpack_params(out["theta"], coord_cfg)
                fourier_B = out.get("fourier_B")
                preds = coord_forward(coords, params, coord_cfg, fourier_B=fourier_B)
                loss = F.mse_loss(preds, targets)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()

            epoch_loss += loss.item() * bsz

        mean_loss = epoch_loss / len(train_loader.dataset)
        elapsed = time.time() - start

        eval_stats = evaluate(
            model=model,
            cfg=coord_cfg,
            loader=test_loader,
            device=device,
            image_size=args.image_size,
            eval_batches=args.eval_batches,
            lpips_fn=lpips_fn,
            superres_size=args.superres_eval_size if args.superres_eval_size > 0 else None,
            superres_lowres=args.train_lowres if args.train_lowres > 0 else None,
        )

        row = {
            "epoch": epoch,
            "train_mse": mean_loss,
            "time_sec": elapsed,
            **eval_stats,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        scheduler.step()

        msg = f"epoch={epoch:03d} train_mse={mean_loss:.6f} lr={scheduler.get_last_lr()[0]:.2e} time={elapsed:.1f}s"
        msg += f" val_psnr={eval_stats['psnr']:.2f} val_ssim={eval_stats['ssim']:.4f}"
        if "lpips" in eval_stats:
            msg += f" val_lpips={eval_stats['lpips']:.4f}"
        print(msg)

        if eval_stats["psnr"] > best_psnr:
            best_psnr = eval_stats["psnr"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "coord_cfg": vars(coord_cfg),
                    "args": vars(args),
                    "best_psnr": best_psnr,
                },
                ckpt_path,
            )
            print(f"saved checkpoint: {ckpt_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hypernetwork -> INR reconstruction training")
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "celeba"])
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--out-dir", type=str, default="./runs")

    p.add_argument("--arch", type=str, default="siren", choices=["siren", "relu", "fourier"])
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--first-omega", type=float, default=30.0)
    p.add_argument("--hidden-omega", type=float, default=30.0)
    p.add_argument("--fourier-features", type=int, default=16)
    p.add_argument("--fourier-sigma", type=float, default=10.0)

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--samples-per-image", type=int, default=1024)
    p.add_argument("--eval-batches", type=int, default=20)

    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--train-lowres", type=int, default=0)
    p.add_argument("--superres-eval-size", type=int, default=0)

    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="Max gradient norm (0 to disable).")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA.")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
