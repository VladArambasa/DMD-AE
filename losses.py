from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from tensor_diagnostics import check_tensor
import defines


def make_reconstruction_loss(
    loss_mode: int = 0,
    beta: float = 0.05,
    eps: float = 1e-6,
    w_pow: float = 1.0,
):
    if loss_mode == 0:
        def loss_fn(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return torch.mean((x_hat - x) ** 2)

        return loss_fn

    if loss_mode == 1:
        def loss_fn(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return torch.mean(torch.abs(x_hat - x))

        return loss_fn

    if loss_mode == 2:
        def loss_fn(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return F.smooth_l1_loss(x_hat, x, beta=beta)

        return loss_fn

    if loss_mode == 3:
        def loss_fn(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            weights = 1.0 / (torch.abs(x) ** w_pow + eps)
            return torch.mean(weights * (x_hat - x) ** 2)

        return loss_fn

    raise ValueError(f"UNKNOWN LOSS MODE: {loss_mode}")


def fit_batch_dmd_matrix_OLD(
    z1: torch.Tensor,
    z2: torch.Tensor,
    ridge: float = 1e-6,
) -> torch.Tensor:
    latent_dim = z1.shape[1]

    eye = torch.eye(
        latent_dim,
        dtype=z1.dtype,
        device=z1.device,
    )

    G = z1.T @ z1
    H = z1.T @ z2

    A_t = torch.linalg.solve(G + ridge * eye, H)
    A = A_t.T

    return A

def fit_batch_dmd_matrix_NEW(
        z1: torch.Tensor,
        z2: torch.Tensor,
        ridge: float = defines.DMD_RIDGE,
) -> torch.Tensor:
    U, S, Vh = torch.linalg.svd(z1, full_matrices=False)
    S_inv = S / (S * S + ridge)
    z1_pinv = (Vh.transpose(-1,-2) * S_inv) @ U.transpose(-1,-2)
    A_t = z1_pinv @ z2
    return A_t.transpose(-1,-2)

def fit_batch_dmd_matrix__CU_CHECKER(
        z1: torch.Tensor,
        z2: torch.Tensor,
        ridge: float = defines.DMD_RIDGE,
) -> torch.Tensor:
    check_tensor(z1, name="z1")
    U, S, Vh = torch.linalg.svd(z1, full_matrices=False)
    S_inv = S / (S * S + ridge)
    z1_pinv = (Vh.transpose(-1,-2) * S_inv) @ U.transpose(-1,-2)
    A_t = z1_pinv @ z2
    return A_t.transpose(-1,-2)

def fit_batch_dmd_matrix(
        z1: torch.Tensor,
        z2: torch.Tensor,
        ridge: float = defines.DMD_RIDGE,
) -> torch.Tensor:

    latent_dim = z1.shape[1]
    eye = torch.eye(latent_dim, dtype=z1.dtype, device=z1.device)
    G = z1.T @ z1
    H = z1.T @ z2
    A_t = torch.linalg.solve(G + ridge * eye, H)
    return A_t.T
def compute_target_scale(*arrays_or_tensors, eps: float = 1e-12) -> float:
    total = 0.0
    count = 0
    for arr in arrays_or_tensors:
        if arr is None:
            continue
        if isinstance(arr, torch.Tensor):
            total += float(torch.sum(arr.detach() ** 2).cpu())
            count += int(arr.numel())
        else:
            a = np.asarray(arr, dtype=np.float64) if not isinstance(arr, np.ndarray) else arr
            total += float(np.sum(a.astype(np.float64) ** 2))
            count += int(a.size)
    if count == 0:
        return 1.0
    return max(total / count, eps)


def koopman_ae_loss(
    x1: torch.Tensor,
    x2: torch.Tensor,
    z1: torch.Tensor,
    z2: torch.Tensor,
    x1_rec: torch.Tensor,
    x2_rec: torch.Tensor,
    decoder,
    base_loss,
    alpha_rec: float = 0.5,
    alpha_lin: float = 0.5,
    alpha_pred: float = 2.0,
    ridge: float = 1e-6,
    scale: float = 1.0,
):

    A = fit_batch_dmd_matrix(z1, z2, ridge=ridge)
    z2_pred = z1 @ A.T
    x2_pred = decoder(z2_pred)

    loss_rec = base_loss(x1_rec, x1) + base_loss(x2_rec, x2)
    loss_lin = base_loss(z2_pred, z2)
    loss_pred = base_loss(x2_pred, x2)

    inv_scale = 1.0 / max(float(scale), 1e-12)

    loss_total = inv_scale * (
        alpha_rec * loss_rec
        + alpha_lin * loss_lin
        + alpha_pred * loss_pred
    )

    loss_info = {
        "total": float(loss_total.detach().cpu()),
        "rec": float(loss_rec.detach().cpu()),
        "lin": float(loss_lin.detach().cpu()),
        "pred": float(loss_pred.detach().cpu()),
        "scale": float(scale),
    }

    return loss_total, loss_info