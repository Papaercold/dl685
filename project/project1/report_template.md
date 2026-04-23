# ECE 685D Final Project Report Template (Project 1)

## 1. Introduction

- Briefly introduce implicit neural representations (INRs) and hypernetworks.
- State your project goals for CIFAR-10 and CelebA.

## 2. Method

### 2.1 INR formulation

Use:

\[
f_\theta: \mathbb{R}^2 \rightarrow \mathbb{R}^3
\]

and reconstruction objective:

\[
\mathcal{L}(\phi)=\mathbb{E}_{I}\mathbb{E}_{(x,y)}\left\|f_{H_\phi(I)}(x,y)-I(x,y)\right\|_2^2.
\]

### 2.2 Architectures

Describe your tested coordinate networks:

- SIREN
- Fourier feature MLP
- ReLU MLP

Describe hypernetwork architecture (CNN encoder + MLP head generating `theta`).

## 3. Experimental Setup

### 3.1 Datasets

- CIFAR-10 (32x32)
- CelebA (resized to 128x128)

### 3.2 Training setup

- Optimizer, LR, batch size, epochs
- Sampled coordinates per image
- Hardware and runtime

### 3.3 Metrics

- PSNR
- SSIM
- LPIPS

## 4. Results

### 4.1 CIFAR-10 reconstruction

| Architecture | PSNR (dB) | SSIM | LPIPS |
|---|---:|---:|---:|
| SIREN |  |  |  |
| Fourier |  |  |  |
| ReLU |  |  |  |

Add qualitative reconstructions (`target` vs `recon`).

### 4.2 Downstream CIFAR-10 classification

| Feature type | Accuracy (%) |
|---|---:|
| Raw pixels |  |
| Hypernetwork latent |  |
| Generated INR weights (theta) |  |

### 4.3 CelebA architecture comparison

| Architecture | PSNR (dB) | SSIM | LPIPS | Train time / epoch |
|---|---:|---:|---:|---:|
| SIREN |  |  |  |  |
| Fourier |  |  |  |  |
| ReLU |  |  |  |  |

### 4.4 Super-resolution test

Train with low-resolution supervision and evaluate at high-resolution.

| Train supervision | Eval resolution | PSNR (dB) | SSIM | LPIPS |
|---|---|---:|---:|---:|
| 64x64 | 256x256 |  |  |  |

## 5. Discussion

- Which INR architecture gives best quality/stability?
- What is the quality-cost tradeoff?
- Are learned representations useful for downstream classification?
- How well does INR generalize to higher resolution?

## 6. Conclusion

- Summarize your main findings.
- Mention limitations and future improvements.
