# ECE 685D Final Project 1: Hypernetworks for INRs

This folder contains the Project 1 implementation and experiment scripts.

## Scope Covered

- **Task I (CIFAR-10):** Hypernetwork-based INR reconstruction + downstream classification.
- **Task II (CelebA):** Architecture comparison (SIREN / Fourier / ReLU), PSNR/SSIM/LPIPS, and super-resolution testing.

## Files

- `hyperinr.py`: Hypernetwork and generated coordinate-network forward pass.
- `data_utils.py`: CIFAR-10/CelebA loaders, coordinate sampling, image resizing.
- `metrics.py`: PSNR, SSIM, optional LPIPS.
- `train_reconstruction.py`: Main training/evaluation script.
- `train_downstream_classifier.py`: CIFAR-10 classifier from hypernetwork features.
- `export_recon_examples.py`: Exports paired target/reconstruction images.
- `run_experiments.sh`: End-to-end command template.
- `project1_runbook.ipynb`: Notebook runbook with all experiment commands.
- `report_template.md`: Report structure template.

## Requirements

- Python 3.10+
- PyTorch + torchvision
- Optional: `lpips` (`pip install lpips`)

If LPIPS is unavailable, training continues with PSNR/SSIM.

## Quick Start

```bash
cd /home/zihan-gao/dl685/project/project1
python train_reconstruction.py \
  --dataset cifar10 --arch siren --image-size 32 \
  --epochs 20 --batch-size 128 --samples-per-image 1024 \
  --num-workers 8 --amp --out-dir ./runs
```

Outputs:

- Checkpoint: `runs/cifar10_siren_recon.pt`
- Metrics log: `runs/cifar10_siren_recon_metrics.jsonl`

## Task I: CIFAR-10 + Downstream Classification

```bash
python train_reconstruction.py --dataset cifar10 --arch siren  --image-size 32 --epochs 120 --batch-size 128 --samples-per-image 4096 --hidden-dim 128 --depth 4 --latent-dim 512 --lr 3e-4 --weight-decay 1e-7 --num-workers 8 --amp --eval-batches 50 --out-dir ./runs
python train_reconstruction.py --dataset cifar10 --arch fourier --image-size 32 --epochs 120 --batch-size 128 --samples-per-image 4096 --hidden-dim 128 --depth 4 --latent-dim 512 --fourier-features 32 --fourier-sigma 10 --lr 3e-4 --weight-decay 1e-7 --num-workers 8 --amp --eval-batches 50 --out-dir ./runs
python train_reconstruction.py --dataset cifar10 --arch relu   --image-size 32 --epochs 120 --batch-size 128 --samples-per-image 4096 --hidden-dim 128 --depth 4 --latent-dim 512 --lr 3e-4 --weight-decay 1e-7 --num-workers 8 --amp --eval-batches 50 --out-dir ./runs
```

```bash
python train_downstream_classifier.py --recon-ckpt ./runs/cifar10_siren_recon.pt --feature-type latent --cls-epochs 40 --out-path ./runs/cifar10_latent_cls.pt
python train_downstream_classifier.py --recon-ckpt ./runs/cifar10_siren_recon.pt --feature-type theta  --cls-epochs 40 --out-path ./runs/cifar10_theta_cls.pt
```

## Task II: CelebA + Super-Resolution

```bash
python train_reconstruction.py --dataset celeba --arch siren  --image-size 128 --epochs 60 --batch-size 32 --samples-per-image 4096 --hidden-dim 128 --depth 4 --latent-dim 512 --lr 2e-4 --weight-decay 1e-7 --num-workers 8 --amp --eval-batches 30 --out-dir ./runs
python train_reconstruction.py --dataset celeba --arch fourier --image-size 128 --epochs 60 --batch-size 32 --samples-per-image 4096 --hidden-dim 128 --depth 4 --latent-dim 512 --fourier-features 32 --fourier-sigma 10 --lr 2e-4 --weight-decay 1e-7 --num-workers 8 --amp --eval-batches 30 --out-dir ./runs
python train_reconstruction.py --dataset celeba --arch relu   --image-size 128 --epochs 60 --batch-size 32 --samples-per-image 4096 --hidden-dim 128 --depth 4 --latent-dim 512 --lr 2e-4 --weight-decay 1e-7 --num-workers 8 --amp --eval-batches 30 --out-dir ./runs
```

```bash
python train_reconstruction.py \
  --dataset celeba --arch siren --image-size 128 \
  --train-lowres 64 --superres-eval-size 256 \
  --epochs 60 --batch-size 32 --samples-per-image 4096 \
  --hidden-dim 128 --depth 4 --latent-dim 512 \
  --lr 2e-4 --weight-decay 1e-7 --num-workers 8 --amp --eval-batches 30 --out-dir ./runs
```

## Export Figures

```bash
python export_recon_examples.py \
  --recon-ckpt ./runs/celeba_siren_recon.pt \
  --num-images 30 --out-dir ./runs/figures
```
