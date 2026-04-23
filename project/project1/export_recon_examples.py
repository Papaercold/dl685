import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from data_utils import build_dataset_loaders, downsample_images, make_grid
from hyperinr import CoordNetConfig, HyperNetwork, coord_forward, total_param_count, unpack_params


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    ckpt = torch.load(args.recon_ckpt, map_location="cpu")
    cfg = CoordNetConfig(**ckpt["coord_cfg"])
    latent_dim = ckpt["args"]["latent_dim"]
    image_size = ckpt["args"]["image_size"]
    train_lowres = args.train_lowres  # 0 means no super-res mode

    output_size = args.output_size if args.output_size > 0 else image_size

    model = HyperNetwork(total_param_count(cfg), latent_dim=latent_dim, coord_cfg=cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    _, test_loader = build_dataset_loaders(
        dataset=ckpt["args"]["dataset"],
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=image_size,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    full_coords = make_grid(output_size, device)
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            bsz = images.size(0)
            coords = full_coords.expand(bsz, -1, -1)

            # In super-res mode, feed the low-res image to the hypernetwork,
            # but query the coordinate network at the higher output resolution.
            if train_lowres > 0:
                model_input = downsample_images(images, train_lowres)
            else:
                model_input = images

            out = model(model_input)
            params = unpack_params(out["theta"], cfg)
            fourier_B = out.get("fourier_B")
            pred = coord_forward(coords, params, cfg, fourier_B=fourier_B)
            pred = pred.view(bsz, output_size, output_size, 3).permute(0, 3, 1, 2)

            # Target: original image resized to output_size for fair comparison
            target = F.interpolate(images, size=(output_size, output_size),
                                   mode="bilinear", align_corners=False)

            for i in range(bsz):
                if saved >= args.num_images:
                    return
                save_image(target[i], out_dir / f"{saved:04d}_target.png")
                save_image(pred[i],   out_dir / f"{saved:04d}_recon.png")
                saved += 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export reconstruction / super-resolution examples")
    p.add_argument("--recon-ckpt", type=str, required=True)
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--out-dir", type=str, default="./runs/recon_examples")
    p.add_argument("--num-images", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--output-size", type=int, default=0,
                   help="Query grid size. Defaults to training image_size.")
    p.add_argument("--train-lowres", type=int, default=0,
                   help="If >0, feed this resolution to the hypernetwork (super-res mode).")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
