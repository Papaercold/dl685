import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import os

nb = new_notebook()
cells = []

# ============================================================
# Title
# ============================================================
cells.append(new_markdown_cell(
    "# ECE 685D HW4: Feature Extraction, Representation Learning, and Score-Based Models\n\n"
    "**Student:** Zihan Gao (zg137)  \n"
    "**Date:** March 2026\n\n---\n"
))

# ============================================================
# Imports
# ============================================================
cells.append(new_code_cell("\n".join([
    "import numpy as np",
    "import torch",
    "import torch.nn as nn",
    "import torch.optim as optim",
    "from torch.utils.data import DataLoader, TensorDataset",
    "import torchvision",
    "import torchvision.transforms as transforms",
    "import matplotlib",
    "matplotlib.use('Agg')",
    "import matplotlib.pyplot as plt",
    "import warnings",
    "warnings.filterwarnings('ignore')",
    "",
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
    "print(f'Using device: {device}')",
])))

# ============================================================
# TASK 1 HEADER
# ============================================================
cells.append(new_markdown_cell(
    "---\n# Task 1: Feature Extraction and Representation Learning (50 pts)\n"
))

# ============================================================
# Task 1(a)
# ============================================================
cells.append(new_markdown_cell(
    r"## Task 1(a): Eckart-Young Theorem (10 pts)" + "\n\n"
    r"**Claim:** Let $A \in \mathbb{R}^{m \times n}$ with SVD $A = U \Sigma V^T$ where "
    r"$\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$. Define the rank-$k$ truncated SVD:"
    "\n$$A_k = \\sum_{i=1}^{k} \\sigma_i u_i v_i^T$$\n"
    r"Then for any $B$ with $\text{rank}(B) \leq k$: $\|A - A_k\|_F \leq \|A - B\|_F$."
    "\n\n**Proof (using the Hoffman-Wielandt inequality):**\n\n"
    "**Step 1: Error of truncated SVD.**\n\n"
    r"Since $A - A_k = \sum_{i=k+1}^{r} \sigma_i u_i v_i^T$, we get:"
    "\n$$\\|A - A_k\\|_F^2 = \\sum_{i=k+1}^{r} \\sigma_i^2$$\n\n"
    "**Step 2: Hoffman-Wielandt inequality.**\n\n"
    r"For any two matrices $X, Y \in \mathbb{R}^{m \times n}$:"
    "\n$$\\sum_{i=1}^{\\min(m,n)} (\\sigma_i(X) - \\sigma_i(Y))^2 \\leq \\|X - Y\\|_F^2$$\n\n"
    r"**Step 3: Apply to $A$ and $B$ with $\text{rank}(B) \leq k$.**" + "\n\n"
    r"Since $\text{rank}(B) \leq k$, we have $\sigma_i(B) = 0$ for $i > k$. By Hoffman-Wielandt:"
    "\n$$\\|A - B\\|_F^2 \\geq \\sum_{i=1}^{r} (\\sigma_i(A) - \\sigma_i(B))^2 "
    "\\geq \\sum_{i=k+1}^{r} \\sigma_i(A)^2 = \\|A - A_k\\|_F^2$$\n\n"
    r"The second inequality holds because we restrict the sum to $i > k$ (all terms nonneg)."
    "\n\n**Step 4: Conclude.**\n\n"
    r"$\|A - B\|_F \geq \|A - A_k\|_F$ for all rank-$k$ matrices $B$. "
    r"The truncated SVD $A_k$ is the optimal rank-$k$ approximation in Frobenius norm. $\blacksquare$"
))

# ============================================================
# Task 1(b)
# ============================================================
cells.append(new_markdown_cell(
    r"## Task 1(b): PCA — Max Variance = Min Reconstruction Error (10 pts)" + "\n\n"
    r"Let $X = [x_1,\ldots,x_N] \in \mathbb{R}^{d\times N}$ be centered data, "
    r"$S = \frac{1}{N}XX^T$ the sample covariance. "
    r"Encoder $W \in \mathbb{R}^{k\times d}$ with $WW^T = I_k$."
    "\n\n**Claim:** Maximizing $\text{tr}(WSW^T)$ and minimizing "
    r"$\frac{1}{N}\sum_n\|x_n - W^TWx_n\|^2$ are equivalent, "
    r"both solved by the top-$k$ eigenvectors of $S$."
    "\n\n**Step 1: Expand reconstruction error.**\n\n"
    "$$\\frac{1}{N}\\sum_n\\|x_n - W^TWx_n\\|^2 = \\frac{1}{N}\\sum_n"
    "\\left(\\|x_n\\|^2 - 2x_n^TW^TWx_n + x_n^T(W^TW)^2x_n\\right)$$\n\n"
    r"Since $WW^T = I_k$, we have $(W^TW)^2 = W^T(WW^T)W = W^TW$, so:"
    "\n$$= \\frac{1}{N}\\sum_n\\|x_n\\|^2 - \\frac{1}{N}\\sum_n x_n^TW^TWx_n"
    " = \\text{const} - \\text{tr}(WSW^T)$$\n\n"
    "**Step 2: Equivalence.** Minimizing $\\text{const} - \\text{tr}(WSW^T)$ "
    "is equivalent to maximizing $\\text{tr}(WSW^T)$.\n\n"
    "**Step 3: Optimal $W$.** By the Ky Fan maximum principle:\n"
    "$$\\max_{WW^T=I_k}\\text{tr}(WSW^T) = \\sum_{i=1}^k\\lambda_i(S)$$\n"
    r"achieved when rows of $W$ are the top-$k$ eigenvectors of $S$. $\blacksquare$"
))

# ============================================================
# Task 1(c)
# ============================================================
cells.append(new_markdown_cell(
    r"## Task 1(c): Non-Uniqueness of Optimal Autoencoder (10 pts)" + "\n\n"
    r"**Claim:** If $(W_{\text{enc}}, W_{\text{dec}})$ is optimal, then "
    r"$(RW_{\text{enc}},\, W_{\text{dec}}R^{-1})$ is also optimal for any invertible $R$."
    "\n\n**Proof:**\n\n"
    r"The reconstruction is $\hat{x} = W_{\text{dec}}W_{\text{enc}}x$. For the transformed pair:"
    "\n$$(W_{\\text{dec}}R^{-1})(RW_{\\text{enc}})x = W_{\\text{dec}}(R^{-1}R)W_{\\text{enc}}x = "
    r"W_{\text{dec}}W_{\text{enc}}x = \hat{x}$$"
    "\n\nSince the reconstruction is unchanged, the MSE is unchanged. $\\blacksquare$\n\n"
    "---\n\n**Connection to PCA:**\n\n"
    "From (b), the optimal linear autoencoder reconstructs via the orthogonal projection onto "
    r"the top-$k$ eigenspace of $S$. This projection $P^* = W_{\text{PCA}}^T W_{\text{PCA}}$ is unique."
    "\n\nHowever, the factorization $(W_{\\text{enc}}, W_{\\text{dec}})$ is not unique: "
    r"any $(RW_{\text{PCA}},\, W_{\text{PCA}}^T R^{-1})$ gives the same $P^*$ since "
    r"$W_{\text{PCA}}^T R^{-1} \cdot R W_{\text{PCA}} = P^*$."
    "\n\n**Conclusion:** The optimal linear autoencoder spans the **same subspace** as PCA "
    "(the top-$k$ eigenspace of $S$), but the individual weight matrices are **not unique** "
    "due to the gauge freedom of multiplying by any invertible $R$."
))

# ============================================================
# Task 1(d) markdown
# ============================================================
cells.append(new_markdown_cell(
    "## Task 1(d): MNIST — PCA vs Linear Autoencoder (10 pts)\n"
))

# Task 1(d) code cells
cells.append(new_code_cell(
    "# Load MNIST\n"
    "transform = transforms.Compose([transforms.ToTensor()])\n"
    "mnist_train = torchvision.datasets.MNIST(root='/tmp/data', train=True,\n"
    "                                          download=True, transform=transform)\n"
    "mnist_test  = torchvision.datasets.MNIST(root='/tmp/data', train=False,\n"
    "                                          download=True, transform=transform)\n"
    "\n"
    "X_train = mnist_train.data.float().reshape(-1, 784) / 255.0\n"
    "X_test  = mnist_test.data.float().reshape(-1, 784)  / 255.0\n"
    "\n"
    "# Center the data\n"
    "mean_train = X_train.mean(dim=0, keepdim=True)\n"
    "X_train_c  = X_train - mean_train\n"
    "X_test_c   = X_test  - mean_train\n"
    "\n"
    "print(f'Train: {X_train_c.shape}, Test: {X_test_c.shape}')\n"
))

cells.append(new_code_cell(
    "# PCA via SVD\n"
    "print('Computing SVD...')\n"
    "U_svd, S_svd, Vh_svd = torch.linalg.svd(X_train_c, full_matrices=False)\n"
    "print(f'SVD shapes: U={U_svd.shape}, S={S_svd.shape}, Vh={Vh_svd.shape}')\n"
    "# Vh_svd[i] is the i-th principal component (row vector)\n"
))

cells.append(new_code_cell(
    "# PCA Reconstruction Error\n"
    "k_values = [1, 2, 5, 10, 20, 50, 100]\n"
    "pca_mse  = []\n"
    "\n"
    "for k in k_values:\n"
    "    W_pca = Vh_svd[:k]\n"
    "    Z      = X_test_c @ W_pca.T\n"
    "    X_rec  = Z @ W_pca + mean_train\n"
    "    mse    = ((X_test - X_rec)**2).mean().item()\n"
    "    pca_mse.append(mse)\n"
    "    print(f'PCA k={k:4d}: MSE = {mse:.6f}')\n"
))

cells.append(new_code_cell(
    "# Linear Autoencoder\n"
    "class LinearAE(nn.Module):\n"
    "    def __init__(self, d, k):\n"
    "        super().__init__()\n"
    "        self.enc = nn.Linear(d, k, bias=False)\n"
    "        self.dec = nn.Linear(k, d, bias=False)\n"
    "\n"
    "    def forward(self, x):\n"
    "        return self.dec(self.enc(x))\n"
    "\n"
    "\n"
    "def train_linear_ae(X_tr, X_te, k, epochs=150, lr=1e-3, batch_size=256):\n"
    "    d = X_tr.shape[1]\n"
    "    model = LinearAE(d, k).to(device)\n"
    "    opt   = optim.Adam(model.parameters(), lr=lr)\n"
    "    loss_fn = nn.MSELoss()\n"
    "    dl = DataLoader(TensorDataset(X_tr.to(device)), batch_size=batch_size, shuffle=True)\n"
    "    for ep in range(epochs):\n"
    "        model.train()\n"
    "        for (xb,) in dl:\n"
    "            opt.zero_grad()\n"
    "            loss_fn(model(xb), xb).backward()\n"
    "            opt.step()\n"
    "    model.eval()\n"
    "    with torch.no_grad():\n"
    "        tl = loss_fn(model(X_te.to(device)), X_te.to(device)).item()\n"
    "    return model, tl\n"
    "\n"
    "\n"
    "ae_mse = []\n"
    "ae_models = {}\n"
    "for k in k_values:\n"
    "    print(f'Training Linear AE k={k}...', end=' ', flush=True)\n"
    "    model, mse = train_linear_ae(X_train_c, X_test_c, k)\n"
    "    ae_mse.append(mse)\n"
    "    ae_models[k] = model\n"
    "    print(f'MSE = {mse:.6f}')\n"
))

cells.append(new_code_cell(
    "# Compare Projection Matrices\n"
    "print('Projection matrix comparison ||P_PCA - P_AE||_F:')\n"
    "for i, k in enumerate(k_values):\n"
    "    W_pca = Vh_svd[:k].cpu()\n"
    "    P_pca = W_pca.T @ W_pca\n"
    "    model = ae_models[k]\n"
    "    Wd = model.dec.weight.data.cpu()  # (d, k)\n"
    "    WdTWd = Wd.T @ Wd\n"
    "    try:\n"
    "        WdTWd_inv = torch.linalg.inv(WdTWd)\n"
    "        P_ae = Wd @ WdTWd_inv @ Wd.T\n"
    "    except Exception:\n"
    "        P_ae = Wd @ torch.linalg.pinv(Wd)\n"
    "    diff = torch.norm(P_pca - P_ae, p='fro').item()\n"
    "    print(f'  k={k:4d}: ||P_PCA - P_AE||_F = {diff:.4f}')\n"
))

cells.append(new_code_cell(
    "# Plot Reconstruction Error vs k\n"
    "fig, ax = plt.subplots(figsize=(8, 5))\n"
    "ax.plot(k_values, pca_mse,  'b-o', label='PCA', linewidth=2)\n"
    "ax.plot(k_values, ae_mse,   'r-s', label='Linear AE', linewidth=2)\n"
    "ax.set_xlabel('Number of components k', fontsize=13)\n"
    "ax.set_ylabel('Reconstruction MSE', fontsize=13)\n"
    "ax.set_title('PCA vs Linear AE: Reconstruction Error on MNIST', fontsize=14)\n"
    "ax.legend(fontsize=12)\n"
    "ax.grid(True, alpha=0.3)\n"
    "plt.tight_layout()\n"
    "plt.savefig('/home/zihan-gao/dl685/homework4/task1d_reconstruction_error.png', dpi=120)\n"
    "plt.show()\n"
    "print('Plot saved.')\n"
))

cells.append(new_code_cell(
    "# Visualise Reconstructions (k=20)\n"
    "k_vis = 20\n"
    "W_pca = Vh_svd[:k_vis]\n"
    "n_show = 8\n"
    "rng = np.random.default_rng(42)\n"
    "idxs = rng.choice(len(X_test), n_show, replace=False)\n"
    "\n"
    "originals = X_test[idxs].numpy()\n"
    "Z_pca = X_test_c[idxs] @ W_pca.T\n"
    "rec_pca = (Z_pca @ W_pca + mean_train).numpy().clip(0, 1)\n"
    "\n"
    "model_vis = ae_models[k_vis]\n"
    "model_vis.eval()\n"
    "with torch.no_grad():\n"
    "    rec_ae = (model_vis(X_test_c[idxs].to(device)).cpu() + mean_train).numpy().clip(0, 1)\n"
    "\n"
    "fig, axes = plt.subplots(3, n_show, figsize=(14, 5))\n"
    "for i in range(n_show):\n"
    "    axes[0, i].imshow(originals[i].reshape(28, 28), cmap='gray'); axes[0, i].axis('off')\n"
    "    axes[1, i].imshow(rec_pca[i].reshape(28, 28),   cmap='gray'); axes[1, i].axis('off')\n"
    "    axes[2, i].imshow(rec_ae[i].reshape(28, 28),    cmap='gray'); axes[2, i].axis('off')\n"
    "axes[0, 0].set_ylabel('Original', fontsize=10)\n"
    "axes[1, 0].set_ylabel('PCA',       fontsize=10)\n"
    "axes[2, 0].set_ylabel('Linear AE', fontsize=10)\n"
    "plt.suptitle(f'MNIST Reconstructions (k={k_vis})', fontsize=13)\n"
    "plt.tight_layout()\n"
    "plt.savefig('/home/zihan-gao/dl685/homework4/task1d_reconstructions.png', dpi=120)\n"
    "plt.show()\n"
))

# ============================================================
# Task 1(e)
# ============================================================
cells.append(new_markdown_cell(
    "## Task 1(e): Nonlinear Autoencoder with MLP (10 pts)\n"
))

cells.append(new_code_cell(
    "class NonlinearAE(nn.Module):\n"
    "    def __init__(self, d=784, k=20):\n"
    "        super().__init__()\n"
    "        self.encoder = nn.Sequential(\n"
    "            nn.Linear(d, 512), nn.ReLU(),\n"
    "            nn.Linear(512, 256), nn.ReLU(),\n"
    "            nn.Linear(256, k)\n"
    "        )\n"
    "        self.decoder = nn.Sequential(\n"
    "            nn.Linear(k, 256), nn.ReLU(),\n"
    "            nn.Linear(256, 512), nn.ReLU(),\n"
    "            nn.Linear(512, d), nn.Sigmoid()\n"
    "        )\n"
    "\n"
    "    def forward(self, x):\n"
    "        return self.decoder(self.encoder(x))\n"
    "\n"
    "\n"
    "def train_nonlinear_ae(X_tr, X_te, k, epochs=50, lr=1e-3, batch_size=256):\n"
    "    model = NonlinearAE(784, k).to(device)\n"
    "    opt   = optim.Adam(model.parameters(), lr=lr)\n"
    "    loss_fn = nn.MSELoss()\n"
    "    dl = DataLoader(TensorDataset(X_tr.to(device)), batch_size=batch_size, shuffle=True)\n"
    "    for ep in range(epochs):\n"
    "        model.train()\n"
    "        for (xb,) in dl:\n"
    "            opt.zero_grad()\n"
    "            loss_fn(model(xb), xb).backward()\n"
    "            opt.step()\n"
    "        if (ep + 1) % 10 == 0:\n"
    "            model.eval()\n"
    "            with torch.no_grad():\n"
    "                tl = loss_fn(model(X_te.to(device)), X_te.to(device)).item()\n"
    "            print(f'  Epoch {ep+1}/{epochs}  test MSE = {tl:.6f}')\n"
    "    model.eval()\n"
    "    with torch.no_grad():\n"
    "        tl = loss_fn(model(X_te.to(device)), X_te.to(device)).item()\n"
    "    return model, tl\n"
    "\n"
    "\n"
    "k_values_nl = [5, 10, 20, 50]\n"
    "nonlinear_mse = {}\n"
    "nonlinear_models = {}\n"
    "\n"
    "for k in k_values_nl:\n"
    "    print(f'Training Nonlinear AE k={k}...')\n"
    "    model, mse = train_nonlinear_ae(X_train, X_test, k, epochs=50)\n"
    "    nonlinear_mse[k] = mse\n"
    "    nonlinear_models[k] = model\n"
    "    print(f'  => Final test MSE = {mse:.6f}')\n"
))

cells.append(new_code_cell(
    "# Compare linear vs nonlinear AE\n"
    "print('Comparison: Linear AE vs Nonlinear MLP AE')\n"
    "print(f'{\"k\":>5} | {\"Linear AE MSE\":>15} | {\"Nonlinear AE MSE\":>16}')\n"
    "print('-' * 42)\n"
    "for k in k_values_nl:\n"
    "    idx = k_values.index(k)\n"
    "    print(f'{k:>5} | {ae_mse[idx]:>15.6f} | {nonlinear_mse[k]:>16.6f}')\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(8, 5))\n"
    "x = np.arange(len(k_values_nl))\n"
    "w = 0.35\n"
    "lin_vals = [ae_mse[k_values.index(k)] for k in k_values_nl]\n"
    "nl_vals  = [nonlinear_mse[k] for k in k_values_nl]\n"
    "ax.bar(x - w/2, lin_vals, w, label='Linear AE', color='steelblue')\n"
    "ax.bar(x + w/2, nl_vals,  w, label='Nonlinear MLP AE', color='tomato')\n"
    "ax.set_xticks(x); ax.set_xticklabels([str(k) for k in k_values_nl])\n"
    "ax.set_xlabel('Bottleneck dimension k', fontsize=13)\n"
    "ax.set_ylabel('Reconstruction MSE', fontsize=13)\n"
    "ax.set_title('Linear vs Nonlinear AE on MNIST', fontsize=14)\n"
    "ax.legend(fontsize=12); ax.grid(True, alpha=0.3, axis='y')\n"
    "plt.tight_layout()\n"
    "plt.savefig('/home/zihan-gao/dl685/homework4/task1e_comparison.png', dpi=120)\n"
    "plt.show()\n"
))

cells.append(new_code_cell(
    "# Visualise Nonlinear AE Reconstructions\n"
    "k_vis = 20\n"
    "n_show = 8\n"
    "rng = np.random.default_rng(42)\n"
    "idxs = rng.choice(len(X_test), n_show, replace=False)\n"
    "\n"
    "originals = X_test[idxs].numpy()\n"
    "model_nl  = nonlinear_models[k_vis]\n"
    "model_nl.eval()\n"
    "with torch.no_grad():\n"
    "    rec_nl = model_nl(X_test[idxs].to(device)).cpu().numpy()\n"
    "\n"
    "W_pca_v = Vh_svd[:k_vis]\n"
    "Z_pca = X_test_c[idxs] @ W_pca_v.T\n"
    "rec_pca = (Z_pca @ W_pca_v + mean_train).numpy().clip(0, 1)\n"
    "\n"
    "fig, axes = plt.subplots(3, n_show, figsize=(14, 5))\n"
    "for i in range(n_show):\n"
    "    axes[0, i].imshow(originals[i].reshape(28, 28), cmap='gray'); axes[0, i].axis('off')\n"
    "    axes[1, i].imshow(rec_pca[i].reshape(28, 28),   cmap='gray'); axes[1, i].axis('off')\n"
    "    axes[2, i].imshow(rec_nl[i].reshape(28, 28),    cmap='gray'); axes[2, i].axis('off')\n"
    "axes[0, 0].set_ylabel('Original',     fontsize=10)\n"
    "axes[1, 0].set_ylabel('PCA',           fontsize=10)\n"
    "axes[2, 0].set_ylabel('Nonlinear MLP', fontsize=10)\n"
    "plt.suptitle(f'Reconstructions: PCA vs Nonlinear AE (k={k_vis})', fontsize=13)\n"
    "plt.tight_layout()\n"
    "plt.savefig('/home/zihan-gao/dl685/homework4/task1e_reconstructions.png', dpi=120)\n"
    "plt.show()\n"
))

# ============================================================
# TASK 2 HEADER
# ============================================================
cells.append(new_markdown_cell(
    "---\n# Task 2: From Energy Models to Score-Based Models (50 pts)\n"
))

# ============================================================
# Task 2(a)
# ============================================================
cells.append(new_markdown_cell(
    r"## Task 2(a): Gaussian-Bernoulli RBM Conditionals (6 pts)" + "\n\n"
    "**Energy function:**\n"
    r"$$E(v,h) = \sum_i \frac{(v_i-b_i)^2}{2\sigma_i^2} - \sum_j \alpha_j h_j"
    r" - \sum_{i,j}\frac{W_{ij}}{\sigma_i}v_i h_j$$"
    "\n\n### (i) Deriving $p(v_i \\mid h)$\n\n"
    r"Collect terms in $v_i$ from $-E(v,h)$:"
    "\n$$-E\\big|_{v_i} = -\\frac{(v_i-b_i)^2}{2\\sigma_i^2} + \\frac{v_i}{\\sigma_i}\\sum_j W_{ij}h_j$$\n\n"
    r"Complete the square. Let $\mu_i = b_i + \sigma_i \sum_j W_{ij}h_j$:"
    "\n$$= -\\frac{(v_i - \\mu_i)^2}{2\\sigma_i^2} + C$$\n\n"
    "Therefore:\n"
    r"$$\boxed{p(v_i\mid h) = \mathcal{N}(\mu_i,\,\sigma_i^2),\quad \mu_i = b_i + \sigma_i\sum_j W_{ij}h_j}$$"
    "\n\n### (ii) Deriving $p(h_j=1 \\mid v)$\n\n"
    r"Terms in $h_j$ from $-E(v,h)$: $\left(\alpha_j + \sum_i \frac{W_{ij}v_i}{\sigma_i}\right)h_j$."
    "\n\nSo $p(h_j=1|v)/p(h_j=0|v) = \\exp\\!\\left(\\alpha_j + \\sum_i W_{ij}v_i/\\sigma_i\\right)$, giving:\n"
    r"$$\boxed{p(h_j=1\mid v) = \sigma\!\left(\alpha_j + \sum_i \frac{W_{ij}v_i}{\sigma_i}\right)}$$"
    "\n\nwhere $\\sigma(\\cdot)$ is the sigmoid function. $\\blacksquare$"
))

# ============================================================
# Task 2(b)
# ============================================================
cells.append(new_markdown_cell(
    "## Task 2(b): Gaussian-Bernoulli RBM on Fashion MNIST (10 pts)\n"
))

cells.append(new_code_cell(
    "# Load Fashion MNIST\n"
    "transform = transforms.Compose([transforms.ToTensor()])\n"
    "fmnist_train = torchvision.datasets.FashionMNIST(root='/tmp/data', train=True,\n"
    "                                                   download=True, transform=transform)\n"
    "fmnist_test  = torchvision.datasets.FashionMNIST(root='/tmp/data', train=False,\n"
    "                                                   download=True, transform=transform)\n"
    "\n"
    "Fv_train = fmnist_train.data.float().reshape(-1, 784) / 255.0\n"
    "Fv_test  = fmnist_test.data.float().reshape(-1, 784)  / 255.0\n"
    "print(f'Fashion MNIST train: {Fv_train.shape}, test: {Fv_test.shape}')\n"
))

cells.append(new_code_cell(
    "class GaussianBernoulliRBM:\n"
    "    def __init__(self, d=784, m=100, lr=0.01, dev='cpu'):\n"
    "        self.d, self.m, self.lr, self.dev = d, m, lr, dev\n"
    "        self.W     = torch.randn(m, d, device=dev) * 0.01\n"
    "        self.b     = torch.zeros(d,    device=dev)\n"
    "        self.alpha = torch.zeros(m,    device=dev)\n"
    "        self.sigma = torch.ones(d,     device=dev)\n"
    "\n"
    "    def p_h_given_v(self, v):\n"
    "        v_norm = v / self.sigma.unsqueeze(0)\n"
    "        pre    = v_norm @ self.W.T + self.alpha\n"
    "        return torch.sigmoid(pre)\n"
    "\n"
    "    def sample_h(self, v):\n"
    "        p = self.p_h_given_v(v)\n"
    "        return (torch.rand_like(p) < p).float()\n"
    "\n"
    "    def mean_v_given_h(self, h):\n"
    "        return self.b + self.sigma * (h @ self.W)\n"
    "\n"
    "    def sample_v(self, h):\n"
    "        mu  = self.mean_v_given_h(h)\n"
    "        return (mu + torch.randn_like(mu) * self.sigma).clamp(0, 1)\n"
    "\n"
    "    def cd_k(self, v0, k=1):\n"
    "        h0_prob = self.p_h_given_v(v0)\n"
    "        h0      = (torch.rand_like(h0_prob) < h0_prob).float()\n"
    "        vk, hk  = v0.clone(), h0.clone()\n"
    "        for _ in range(k):\n"
    "            vk = self.sample_v(hk)\n"
    "            hk_prob = self.p_h_given_v(vk)\n"
    "            hk = (torch.rand_like(hk_prob) < hk_prob).float()\n"
    "        hk_prob_f = self.p_h_given_v(vk)\n"
    "        sig2 = self.sigma.pow(2)\n"
    "        dW     = (h0_prob.T @ (v0/sig2) - hk_prob_f.T @ (vk/sig2)) / v0.shape[0]\n"
    "        db     = ((v0 - vk)/sig2).mean(0)\n"
    "        dalpha = (h0_prob - hk_prob_f).mean(0)\n"
    "        return dW, db, dalpha\n"
    "\n"
    "    def update(self, dW, db, da):\n"
    "        self.W     += self.lr * dW\n"
    "        self.b     += self.lr * db\n"
    "        self.alpha += self.lr * da\n"
    "\n"
    "    def reconstruct(self, v):\n"
    "        h = self.sample_h(v)\n"
    "        return self.mean_v_given_h(h)\n"
    "\n"
    "    def recon_mse(self, v):\n"
    "        return ((v - self.reconstruct(v))**2).mean().item()\n"
    "\n"
    "\n"
    "def train_rbm(d, m, Xtr, Xte, epochs=25, bs=128, lr=0.01):\n"
    "    rbm = GaussianBernoulliRBM(d=d, m=m, lr=lr, dev=device)\n"
    "    Xt  = Xtr.to(device)\n"
    "    Xte = Xte.to(device)\n"
    "    dl  = DataLoader(TensorDataset(Xt), batch_size=bs, shuffle=True)\n"
    "    for ep in range(1, epochs+1):\n"
    "        for (vb,) in dl:\n"
    "            dW, db, da = rbm.cd_k(vb, k=1)\n"
    "            rbm.update(dW, db, da)\n"
    "        if ep % 5 == 0 or ep == 1:\n"
    "            mse = rbm.recon_mse(Xte[:1000])\n"
    "            print(f'  m={m:4d}  Epoch {ep:3d}/{epochs}  MSE={mse:.5f}')\n"
    "    return rbm, rbm.recon_mse(Xte)\n"
    "\n"
    "\n"
    "hidden_dims = [10, 50, 100, 250]\n"
    "rbm_results = {}\n"
    "for m in hidden_dims:\n"
    "    print(f'--- RBM m={m} ---')\n"
    "    rbm, mse = train_rbm(784, m, Fv_train, Fv_test)\n"
    "    rbm_results[m] = {'rbm': rbm, 'mse': mse}\n"
    "    print(f'  => Final MSE = {mse:.5f}')\n"
    "\n"
    "print('\\nSummary:')\n"
    "for m in hidden_dims:\n"
    "    print(f'  m={m:4d}: MSE={rbm_results[m][\"mse\"]:.5f}')\n"
))

cells.append(new_code_cell(
    "# Plot 16 reconstructions for m=100\n"
    "rbm100 = rbm_results[100]['rbm']\n"
    "rng = np.random.default_rng(0)\n"
    "idxs = rng.choice(len(Fv_test), 16, replace=False)\n"
    "v_orig = Fv_test[idxs].to(device)\n"
    "v_rec  = rbm100.reconstruct(v_orig).cpu().detach()\n"
    "\n"
    "fig, axes = plt.subplots(4, 8, figsize=(14, 7))\n"
    "for i in range(16):\n"
    "    r, c = (i // 8) * 2, i % 8\n"
    "    axes[r,   c].imshow(v_orig[i].cpu().numpy().reshape(28,28), cmap='gray')\n"
    "    axes[r,   c].axis('off')\n"
    "    axes[r+1, c].imshow(v_rec[i].numpy().reshape(28,28), cmap='gray')\n"
    "    axes[r+1, c].axis('off')\n"
    "fig.text(0.01, 0.75, 'Original',      va='center', rotation='vertical', fontsize=11)\n"
    "fig.text(0.01, 0.25, 'Reconstructed', va='center', rotation='vertical', fontsize=11)\n"
    "plt.suptitle('RBM (m=100) Reconstructions on Fashion MNIST', fontsize=13)\n"
    "plt.tight_layout()\n"
    "plt.savefig('/home/zihan-gao/dl685/homework4/task2b_rbm_reconstructions.png', dpi=120)\n"
    "plt.show()\n"
))

# ============================================================
# Task 2(c)
# ============================================================
cells.append(new_markdown_cell(
    r"## Task 2(c): Score Matching Derivation (12 pts)" + "\n\n"
    "### (c1): Implicit Score Matching Objective\n\n"
    r"**Claim:** $J(\theta) = \mathbb{E}_{p_{\text{data}}}\!\left[\frac{1}{2}\|s_\theta(x)\|^2 + \operatorname{div}(s_\theta(x))\right] + C$"
    "\n\nStart from the explicit score matching objective:\n"
    r"$$J(\theta) = \mathbb{E}\!\left[\tfrac{1}{2}\|s_\theta(x)-\nabla_x\log p_{\text{data}}(x)\|^2\right]$$"
    "\n\n**Expand:**\n"
    r"$$= \mathbb{E}\!\left[\tfrac{1}{2}\|s_\theta\|^2\right]"
    r" - \mathbb{E}\!\left[s_\theta(x)^T\nabla_x\log p_{\text{data}}(x)\right]"
    r" + \underbrace{\mathbb{E}\!\left[\tfrac{1}{2}\|\nabla_x\log p_{\text{data}}\|^2\right]}_{C}$$"
    "\n\n**Integration by parts on the cross term** (component $j$):\n"
    r"$$\mathbb{E}\!\left[s_{\theta,j}\,\partial_j\log p\right]"
    r"= \int s_{\theta,j}(x)\,\partial_j p_{\text{data}}(x)\,dx"
    r"= -\int \partial_j s_{\theta,j}(x)\,p_{\text{data}}(x)\,dx"
    r"= -\mathbb{E}\!\left[\partial_j s_{\theta,j}\right]$$"
    "\n\n(boundary terms vanish since $p_{\\text{data}} \\to 0$ at infinity)\n\n"
    "**Summing over $j$:**\n"
    r"$$\mathbb{E}\!\left[s_\theta^T\nabla\log p_{\text{data}}\right] = -\mathbb{E}\!\left[\operatorname{div}(s_\theta)\right]$$"
    "\n\n**Therefore:**\n"
    r"$$\boxed{J(\theta) = \mathbb{E}\!\left[\tfrac{1}{2}\|s_\theta(x)\|^2 + \operatorname{div}(s_\theta(x))\right] + C}$$"
    "\n$\\blacksquare$\n\n"
    "---\n"
    "### (c2): Score of EBM Does Not Depend on $Z(\\theta)$\n\n"
    r"For $p_\theta(x) = \exp(-E_\theta(x))/Z(\theta)$:"
    "\n$$s_\\theta(x) = \\nabla_x\\log p_\\theta(x) = \\nabla_x[-E_\\theta(x) - \\log Z(\\theta)]"
    r" = -\nabla_x E_\theta(x)$$"
    "\n\nsince $\\log Z(\\theta)$ does not depend on $x$. Hence:\n"
    r"$$\boxed{s_\theta(x) = -\nabla_x E_\theta(x)}$$"
    "\n\nThe score matching objective $J(\\theta)$ involves only $s_\\theta$ and its divergence, "
    "both computable without $Z(\\theta)$. Score matching provides a **tractable training objective** "
    "for EBMs. $\\blacksquare$"
))

# ============================================================
# Task 2(d)
# ============================================================
cells.append(new_markdown_cell(
    r"## Task 2(d): Denoising Score Matching Equivalence (10 pts)" + "\n\n"
    r"**Setup.** $X \sim p_{\text{data}}$, $\varepsilon \sim \mathcal{N}(0,\sigma^2 I)$, "
    r"$\tilde{X} = X + \varepsilon$. Smoothed density: $p_\sigma(\tilde{x}) = \int p_{\text{data}}(x)\mathcal{N}(\tilde{x};x,\sigma^2 I)\,dx$."
    "\n\n**DSM objective:**\n"
    r"$$J_{\text{DSM}}(\theta) = \mathbb{E}_{X,\varepsilon}\!\left[\left\|s_\theta(X+\varepsilon) + \frac{\varepsilon}{\sigma^2}\right\|^2\right]$$"
    "\n\n**Expand:**\n"
    r"$$= \mathbb{E}\!\left[\|s_\theta(\tilde{X})\|^2\right] + \frac{2}{\sigma^2}\mathbb{E}\!\left[s_\theta(\tilde{X})^T\varepsilon\right] + \underbrace{\mathbb{E}\!\left[\|\varepsilon/\sigma^2\|^2\right]}_{C}$$"
    "\n\n**Key identity (from the hint):**\n"
    r"$$\nabla_{\tilde{x}}\log p_\sigma(\tilde{x}) = \mathbb{E}\!\left[-\frac{\varepsilon}{\sigma^2}\,\Big|\,\tilde{X}=\tilde{x}\right]$$"
    "\n\n*Proof:* $\\nabla_{\\tilde{x}}\\log p_\\sigma(\\tilde{x}) = "
    "\\mathbb{E}_X[\\nabla_{\\tilde{x}}\\log\\mathcal{N}(\\tilde{x};X,\\sigma^2 I)\\,|\\,\\tilde{x}] = "
    "\\mathbb{E}[-(\\tilde{x}-X)/\\sigma^2\\,|\\,\\tilde{x}] = \\mathbb{E}[-\\varepsilon/\\sigma^2\\,|\\,\\tilde{X}=\\tilde{x}]$. $\\square$\n\n"
    "**Rewrite cross term:**\n"
    r"$$\frac{1}{\sigma^2}\mathbb{E}\!\left[s_\theta(\tilde{X})^T\varepsilon\right] = \mathbb{E}_{\tilde{X}}\!\left[s_\theta(\tilde{X})^T\mathbb{E}[\varepsilon/\sigma^2\,|\,\tilde{X}]\right] = -\mathbb{E}\!\left[s_\theta(\tilde{X})^T\nabla_{\tilde{x}}\log p_\sigma(\tilde{X})\right]$$"
    "\n\n**Substitute back:**\n"
    r"$$J_{\text{DSM}}(\theta) = \mathbb{E}\!\left[\|s_\theta(\tilde{X})\|^2\right] - 2\mathbb{E}\!\left[s_\theta(\tilde{X})^T\nabla\log p_\sigma\right] + C$$"
    "\n"
    r"$$= \mathbb{E}\!\left[\|s_\theta(\tilde{X}) - \nabla_{\tilde{x}}\log p_\sigma(\tilde{X})\|^2\right] - \underbrace{\mathbb{E}\!\left[\|\nabla\log p_\sigma\|^2\right]}_{\text{const w.r.t. }\theta} + C$$"
    "\n\n**Conclusion:**\n"
    r"$$\boxed{J_{\text{DSM}}(\theta) = \mathbb{E}\!\left[\|s_\theta(\tilde{X})-\nabla_{\tilde{x}}\log p_\sigma(\tilde{X})\|^2\right] + \text{const}}$$"
    "\n\nThe DSM objective and the score matching against $\\nabla\\log p_\\sigma$ are equivalent "
    "up to a constant independent of $\\theta$. $\\blacksquare$"
))

# ============================================================
# Task 2(e)
# ============================================================
cells.append(new_markdown_cell(
    "## Task 2(e): Score Network + Langevin Dynamics Sampling (12 pts)\n"
))

cells.append(new_code_cell(
    "class ScoreNetwork(nn.Module):\n"
    "    def __init__(self, d=784, hidden=1024):\n"
    "        super().__init__()\n"
    "        self.net = nn.Sequential(\n"
    "            nn.Linear(d, hidden),\n"
    "            nn.LayerNorm(hidden),\n"
    "            nn.ReLU(),\n"
    "            nn.Linear(hidden, hidden),\n"
    "            nn.LayerNorm(hidden),\n"
    "            nn.ReLU(),\n"
    "            nn.Linear(hidden, d),\n"
    "        )\n"
    "\n"
    "    def forward(self, x):\n"
    "        return self.net(x)\n"
    "\n"
    "\n"
    "def dsm_loss(model, x, sigma_min=0.1, sigma_max=0.5):\n"
    "    sigma   = torch.empty(x.shape[0], 1, device=x.device).uniform_(sigma_min, sigma_max)\n"
    "    eps     = torch.randn_like(x) * sigma\n"
    "    x_tilde = x + eps\n"
    "    pred    = model(x_tilde)\n"
    "    target  = -eps / (sigma ** 2)\n"
    "    return ((pred - target) ** 2).sum(dim=-1).mean()\n"
    "\n"
    "\n"
    "def train_score_net(Xtr, epochs=40, bs=256, lr=1e-3):\n"
    "    model = ScoreNetwork().to(device)\n"
    "    opt   = optim.Adam(model.parameters(), lr=lr)\n"
    "    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)\n"
    "    dl    = DataLoader(TensorDataset(Xtr.to(device)), batch_size=bs, shuffle=True)\n"
    "    losses = []\n"
    "    for ep in range(1, epochs+1):\n"
    "        model.train()\n"
    "        ep_loss = 0.0\n"
    "        for (xb,) in dl:\n"
    "            opt.zero_grad()\n"
    "            loss = dsm_loss(model, xb)\n"
    "            loss.backward()\n"
    "            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n"
    "            opt.step()\n"
    "            ep_loss += loss.item()\n"
    "        sched.step()\n"
    "        avg = ep_loss / len(dl)\n"
    "        losses.append(avg)\n"
    "        if ep % 5 == 0 or ep == 1:\n"
    "            print(f'  Epoch {ep:3d}/{epochs}  DSM loss = {avg:.4f}')\n"
    "    return model, losses\n"
    "\n"
    "\n"
    "print('Training score network on Fashion MNIST...')\n"
    "score_net, dsm_losses = train_score_net(Fv_train, epochs=40)\n"
    "print('Training complete.')\n"
))

cells.append(new_code_cell(
    "# Training loss plot\n"
    "fig, ax = plt.subplots(figsize=(7, 4))\n"
    "ax.plot(dsm_losses, 'b-', linewidth=2)\n"
    "ax.set_xlabel('Epoch', fontsize=12)\n"
    "ax.set_ylabel('DSM Loss', fontsize=12)\n"
    "ax.set_title('Score Network Training Loss', fontsize=13)\n"
    "ax.grid(True, alpha=0.3)\n"
    "plt.tight_layout()\n"
    "plt.savefig('/home/zihan-gao/dl685/homework4/task2e_training_loss.png', dpi=120)\n"
    "plt.show()\n"
))

cells.append(new_code_cell(
    "@torch.no_grad()\n"
    "def langevin_dynamics(score_model, n=16, T=2000, eta=5e-5):\n"
    "    score_model.eval()\n"
    "    x = torch.randn(n, 784, device=device)\n"
    "    for t in range(T):\n"
    "        score = score_model(x)\n"
    "        x     = x + eta * score + (2*eta)**0.5 * torch.randn_like(x)\n"
    "        if t % 500 == 499:\n"
    "            x = x.clamp(-0.5, 1.5)\n"
    "    return x.clamp(0, 1)\n"
    "\n"
    "\n"
    "print('Running Langevin dynamics (T=2000 steps)...')\n"
    "samples = langevin_dynamics(score_net, n=16, T=2000, eta=5e-5)\n"
    "samples_np = samples.cpu().numpy()\n"
    "print(f'Generated {samples_np.shape[0]} samples.')\n"
))

cells.append(new_code_cell(
    "# Visualise Generated Samples\n"
    "fig, axes = plt.subplots(4, 4, figsize=(8, 8))\n"
    "for i, ax in enumerate(axes.flat):\n"
    "    ax.imshow(samples_np[i].reshape(28, 28), cmap='gray', vmin=0, vmax=1)\n"
    "    ax.axis('off')\n"
    "plt.suptitle('Samples via Langevin Dynamics (Score Network, Fashion MNIST)', fontsize=12)\n"
    "plt.tight_layout()\n"
    "plt.savefig('/home/zihan-gao/dl685/homework4/task2e_generated_samples.png', dpi=120)\n"
    "plt.show()\n"
))

cells.append(new_code_cell(
    "# Real Fashion MNIST for comparison\n"
    "rng = np.random.default_rng(1)\n"
    "idxs = rng.choice(len(Fv_test), 16, replace=False)\n"
    "real_imgs = Fv_test[idxs].numpy()\n"
    "\n"
    "fig, axes = plt.subplots(4, 4, figsize=(8, 8))\n"
    "for i, ax in enumerate(axes.flat):\n"
    "    ax.imshow(real_imgs[i].reshape(28, 28), cmap='gray', vmin=0, vmax=1)\n"
    "    ax.axis('off')\n"
    "plt.suptitle('Real Fashion MNIST Images (for comparison)', fontsize=12)\n"
    "plt.tight_layout()\n"
    "plt.savefig('/home/zihan-gao/dl685/homework4/task2e_real_samples.png', dpi=120)\n"
    "plt.show()\n"
))

# ============================================================
# Summary
# ============================================================
cells.append(new_markdown_cell(
    "---\n## Summary\n\n"
    "| Task | Description | Status |\n"
    "|------|-------------|--------|\n"
    "| 1(a) | Eckart-Young theorem via Hoffman-Wielandt | Proved |\n"
    "| 1(b) | PCA: max variance = min reconstruction | Proved |\n"
    "| 1(c) | Autoencoder non-uniqueness via invertible R | Proved |\n"
    "| 1(d) | MNIST: PCA vs Linear AE comparison | Implemented |\n"
    "| 1(e) | Nonlinear MLP autoencoder | Implemented |\n"
    "| 2(a) | Gaussian-Bernoulli RBM conditionals | Derived |\n"
    "| 2(b) | RBM training on Fashion MNIST (m in {10,50,100,250}) | Implemented |\n"
    "| 2(c) | Score matching via integration by parts + EBM | Derived |\n"
    "| 2(d) | DSM equivalence proof | Proved |\n"
    "| 2(e) | Score network (MLP) + Langevin dynamics | Implemented |\n"
))

# ============================================================
# Assemble and write
# ============================================================
nb.cells = cells
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.10.0"
    }
}

os.makedirs('/home/zihan-gao/dl685/homework4', exist_ok=True)
out_path = '/home/zihan-gao/dl685/homework4/hw4_zg137.ipynb'
with open(out_path, 'w') as f:
    nbformat.write(nb, f)

print(f"Notebook written to: {out_path}")
print(f"File size: {os.path.getsize(out_path) / 1024:.1f} KB")
print(f"Number of cells: {len(nb.cells)}")
