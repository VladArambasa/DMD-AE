from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

import defines as D

from encoder import Encoder
from decoder import Decoder
from quadratic_predictor import make_quadratic_predictor_pair, plot_learned_vs_true_matrix_spectrum
from losses import make_reconstruction_loss
from utils import to_tensor

from data_loader import load_all_A_matrices, split_explicit_matrix_indices, load_one_A_matrix
from prepare_training_data import (
    build_matrix_c_grid_training_data,
    build_matrix_c_grid_training_data_many_matrices,
    save_training_npz,
)
from apply_dmd import fit_dmd_from_latent_covariances
from eval_matrix_dmd_ae import (
    save_loss_curve,
    save_ground_truth_final_mask,
    save_ground_truth_escape_iters,
    iterate_true_next_snapshot,
    next_step_prediction_metrics,
)
from mandelbrot_reconstruct import save_final_snapshot_image, save_escape_image
from experiment_common import (
    make_out_dirs,
    print_metric_block,
    mean_metric_dict,
    write_metrics_txt,
)


# METRICS
def _rel_l2(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.linalg.norm(pred - true) / max(np.linalg.norm(true), 1e-12))


def _mse(pred: np.ndarray, true: np.ndarray) -> float:
    err = pred - true
    return float(np.mean(err * err))


# IMAGE HELPERS
def _save_final_mask_95(Z_final: np.ndarray, escape_r: float, out_png, scale: int = 64) -> str:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    d = int(Z_final.shape[-1] // 2)
    zr = Z_final[..., 0:d].astype(np.float32, copy=False)
    zi = Z_final[..., d:2 * d].astype(np.float32, copy=False)
    r2 = float(escape_r) * float(escape_r)

    comp_mag2 = zr * zr + zi * zi
    max_mag2 = np.max(comp_mag2, axis=-1)
    mask = max_mag2 < r2

    img = (mask.astype(np.uint8) * 255)
    pil = Image.fromarray(img, mode="L")
    pil = pil.resize(
        (img.shape[1] * int(scale), img.shape[0] * int(scale)),
        resample=Image.NEAREST,
    )
    pil.save(out_png)
    return str(out_png)

def _save_mask_from_escape_iters(escape_iters: np.ndarray, max_iters: int, out_png, scale: int = 1) -> str:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    mask = escape_iters.astype(np.int32) >= int(max_iters)
    img = (mask.astype(np.uint8) * 255)

    pil = Image.fromarray(img, mode="L")
    if int(scale) > 1:
        pil = pil.resize(
            (img.shape[1] * int(scale), img.shape[0] * int(scale)),
            resample=Image.NEAREST,
        )

    pil.save(out_png)
    return str(out_png)

def save_escape_image_contrast(escape_iters: np.ndarray, *, out_png) -> str:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    esc = escape_iters.astype(np.float32)
    lo = float(np.min(esc))
    hi = float(np.max(esc))

    if hi <= lo + 1e-6:
        img = np.zeros_like(esc, dtype=np.uint8)
    else:
        norm = (esc - lo) / (hi - lo)
        img = (255.0 * (1.0 - norm)).clip(0, 255).astype(np.uint8)

    Image.fromarray(img, mode="L").save(out_png)
    return str(out_png)


# HELPER FOR PREDICT - ONLY ALIVE ROWS - SE * CEVA LA EA
def _alive_rows(X: np.ndarray, escape_r: float) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    feat_dim = int(X.shape[1])
    d = int((feat_dim - 2) // 2)
    r2 = float(escape_r) * float(escape_r)

    zr = X[:, 0:d]
    zi = X[:, d:2 * d]

    mag2 = zr * zr + zi * zi
    max_mag2 = np.max(np.where(np.isfinite(mag2), mag2, np.inf), axis=1)

    finite = np.isfinite(zr).all(axis=1) & np.isfinite(zi).all(axis=1)
    alive = finite & (max_mag2 < 0.999 * r2)

    return alive
#AE PREDICTOR TRAINER
def train_autoencoder_predict_next_only(
    X1, X2, *, latent_dim: int, epochs: int, batch_size: int, lr: float, device: torch.device,
    encoder: nn.Module | None = None,
    decoder: nn.Module | None = None,
) -> tuple[nn.Module, nn.Module, list[float]]:
    X1_list = [X1] if isinstance(X1, np.ndarray) else list(X1)
    X2_list = [X2] if isinstance(X2, np.ndarray) else list(X2)

    if len(X1_list) == 0 or len(X2_list) == 0:
        raise ValueError("AE PREDICTOR RECEIVED EMPTY X1 OR X2 LIST")
    if len(X1_list) != len(X2_list):
        raise ValueError("AE PREDICTOR X1 AND X2 LISTS MUST HAVE SAME LENGTH")

    in_dim = int(X1_list[0].shape[1])
    enc = (encoder if encoder is not None else Encoder(in_dim, latent_dim)).to(device)
    dec = (decoder if decoder is not None else Decoder(latent_dim, in_dim)).to(device)

    loss_fn = make_reconstruction_loss(0, beta=0.01, w_pow=1.0)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr)




    if bool(getattr(D, "AE_LOSS_AUTO_SCALE", True)):
        from losses import compute_target_scale
        loss_scale = compute_target_scale(*X1_list, *X2_list)
    else:
        loss_scale = 1.0
    inv_scale = 1.0 / max(float(loss_scale), 1e-12)
    print(f"[AE PREDICTOR] loss_scale={loss_scale:.6e}")

    scheduler = None
    if bool(getattr(D, "AE_USE_LR_SCHEDULER", False)):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min",
            patience=int(getattr(D, "AE_LR_PATIENCE", 5)),
            factor=float(getattr(D, "AE_LR_FACTOR", 0.5)),
            min_lr=float(getattr(D, "AE_LR_MIN", 1e-6)),
        )

    losses: list[float] = []

    for e in range(int(epochs)):
        rng = np.random.default_rng(e)
        order = rng.permutation(len(X1_list))

        s = 0.0
        n_batches = 0

        for block_id in order:
            X1_block = np.asarray(X1_list[int(block_id)], dtype=np.float32)
            X2_block = np.asarray(X2_list[int(block_id)], dtype=np.float32)
            if bool(getattr(D, "AE_PRED_USE_ALIVE_ONLY", True)):
                keep = _alive_rows(X1_block, float(D.ESCAPE_R))

                if int(np.count_nonzero(keep)) > 0:
                    X1_block = X1_block[keep].astype(np.float32)
                    X2_block = X2_block[keep].astype(np.float32)
            X1_cpu = torch.tensor(X1_block, dtype=torch.float32)
            X2_cpu = torch.tensor(X2_block, dtype=torch.float32)

            loader = DataLoader(
                TensorDataset(X1_cpu, X2_cpu),
                batch_size=int(batch_size),
                shuffle=True,
                drop_last=False,
            )

            for b1_cpu, b2_cpu in loader:
                b1 = b1_cpu.to(device)
                b2 = b2_cpu.to(device)

                z1 = enc(b1)
                x2_pred = dec(z1)
                loss = inv_scale * loss_fn(x2_pred, b2)

                opt.zero_grad()
                loss.backward()
                opt.step()

                s += float(loss.detach().cpu().item())
                n_batches += 1

        s /= max(1, n_batches)
        losses.append(s)

        if scheduler is not None:
            lr_before = opt.param_groups[0]["lr"]
            scheduler.step(s)
            lr_after = opt.param_groups[0]["lr"]
            if lr_after < lr_before:
                print(f"  [LR SCHEDULER] dropped {lr_before:.3e} -> {lr_after:.3e}")

        print(f"[AE PREDICTOR] EPOCH {e + 1}/{epochs} LOSS={s:.6e}")

    return enc, dec, losses


@torch.no_grad()
def ae_predictor_one_step_metrics(encoder, decoder, X1: np.ndarray, X2: np.ndarray, device) -> dict:
    x1 = to_tensor(X1.astype(np.float32), device)
    pred = decoder(encoder(x1))
    pred_np = pred.detach().cpu().numpy().astype(np.float32)

    if pred_np.shape[1] >= 4 and pred_np.shape[1] == X2.shape[1]:
        pred_np[:, -2:] = X2[:, -2:].astype(np.float32)

    err = pred_np - X2.astype(np.float32)
    mse = float(np.mean(err * err))
    rel_l2 = float(np.linalg.norm(err) / max(np.linalg.norm(X2), 1e-12))
    fit = float(1.0 - rel_l2)
    return {"ae_pred_mse": mse, "ae_pred_rel_l2": rel_l2, "ae_pred_fit": fit}


#AE OUTPUT SNAPSHOTS
@torch.no_grad()
def reconstruct_final_snapshot_ae_only(td, encoder, decoder, device, batch_size: int = 50000) -> np.ndarray:
    if td.X_grid is None:
        raise ValueError("AE ONLY FINAL SNAPSHOT NEEDS td.X_grid")

    feat_dim = int(td.X_grid.shape[-1])
    d = int((feat_dim - 2) // 2)
    H = int(td.X_grid.shape[1])
    W = int(td.X_grid.shape[2])

    X_final = td.X_grid[-1].reshape(-1, feat_dim).astype(np.float32)
    C = X_final[:, 2 * d:2 * d + 2].copy().astype(np.float32)
    X_out = np.zeros((X_final.shape[0], 2 * d), dtype=np.float32)

    for i0 in range(0, X_final.shape[0], int(batch_size)):
        i1 = min(X_final.shape[0], i0 + int(batch_size))
        x = to_tensor(X_final[i0:i1], device)
        x_rec = decoder(encoder(x))
        x_rec[:, 2 * d] = to_tensor(C[i0:i1, 0], device)
        x_rec[:, 2 * d + 1] = to_tensor(C[i0:i1, 1], device)
        x_np = x_rec.detach().cpu().numpy().astype(np.float32)
        X_out[i0:i1, 0:d] = x_np[:, 0:d]
        X_out[i0:i1, d:2 * d] = x_np[:, d:2 * d]

    return X_out.reshape(H, W, 2 * d)


@torch.no_grad()
def predict_next_snapshot_ae_predictor(td, encoder, decoder, device, *, steps: int, escape_r: float, batch_size: int = 50000) -> np.ndarray:

    if td.X_grid is None:
        raise ValueError("AE PREDICT NEXT NEEDS td.X_grid")

    feat_dim = int(td.X_grid.shape[-1])
    d = int((feat_dim - 2) // 2)
    H = int(td.X_grid.shape[1])
    W = int(td.X_grid.shape[2])
    r = float(escape_r)

    X_T = td.X_grid[-1].reshape(-1, feat_dim).astype(np.float32)
    C = X_T[:, 2 * d:2 * d + 2].copy().astype(np.float32)
    X_out = np.zeros((X_T.shape[0], 2 * d), dtype=np.float32)
    n_roll = max(int(steps), 0)

    for i0 in range(0, X_T.shape[0], int(batch_size)):
        i1 = min(X_T.shape[0], i0 + int(batch_size))
        X_t = to_tensor(X_T[i0:i1], device)
        C_t = to_tensor(C[i0:i1], device)

        for _ in range(n_roll):
            X_t = decoder(encoder(X_t))
            X_t[:, 2 * d] = C_t[:, 0]
            X_t[:, 2 * d + 1] = C_t[:, 1]

            x_np = X_t.detach().cpu().numpy().astype(np.float32)
            zr = x_np[:, 0:d]
            zi = x_np[:, d:2 * d]
            mag = np.sqrt(np.maximum(zr * zr + zi * zi, 1e-30)).astype(np.float32)
            bad = (~np.isfinite(mag)) | (mag > r)
            if np.any(bad):
                safe = np.where((mag > 0.0) & np.isfinite(mag), mag, 1.0).astype(np.float32)
                scale = (r / safe).astype(np.float32)
                x_np[:, 0:d] = np.where(bad, zr * scale, zr).astype(np.float32)
                x_np[:, d:2 * d] = np.where(bad, zi * scale, zi).astype(np.float32)
            X_t = to_tensor(x_np, device)

        x_final = X_t.detach().cpu().numpy().astype(np.float32)
        X_out[i0:i1, 0:d] = x_final[:, 0:d]
        X_out[i0:i1, d:2 * d] = x_final[:, d:2 * d]

    return X_out.reshape(H, W, 2 * d)

@torch.no_grad()
def predict_future_snapshots_ae_predictor(
    td,
    encoder,
    decoder,
    device,
    *,
    steps: int,
    escape_r: float,
    batch_size: int = 50000,
) -> np.ndarray:

    if td.X_grid is None:
        raise ValueError("AE FUTURE PREDICT NEEDS td.X_grid")

    feat_dim = int(td.X_grid.shape[-1])
    d = int((feat_dim - 2) // 2)
    H = int(td.X_grid.shape[1])
    W = int(td.X_grid.shape[2])
    r = float(escape_r)
    n_roll = max(int(steps), 0)

    X_N = td.X_grid[-1].reshape(-1, feat_dim).astype(np.float32)
    C = X_N[:, 2 * d:2 * d + 2].copy().astype(np.float32)

    P = int(X_N.shape[0])
    future = np.zeros((n_roll, P, 2 * d), dtype=np.float32)

    for i0 in range(0, P, int(batch_size)):
        i1 = min(P, i0 + int(batch_size))

        X_t = to_tensor(X_N[i0:i1], device)
        C_t = to_tensor(C[i0:i1], device)

        for s in range(n_roll):
            X_t = decoder(encoder(X_t))
            X_t[:, 2 * d] = C_t[:, 0]
            X_t[:, 2 * d + 1] = C_t[:, 1]

            x_np = X_t.detach().cpu().numpy().astype(np.float32)

            zr = x_np[:, 0:d]
            zi = x_np[:, d:2 * d]
            mag = np.sqrt(np.maximum(zr * zr + zi * zi, 1e-30)).astype(np.float32)
            bad = (~np.isfinite(mag)) | (mag > r)

            if np.any(bad):
                safe = np.where((mag > 0.0) & np.isfinite(mag), mag, 1.0).astype(np.float32)
                scale = (r / safe).astype(np.float32)
                x_np[:, 0:d] = np.where(bad, zr * scale, zr).astype(np.float32)
                x_np[:, d:2 * d] = np.where(bad, zi * scale, zi).astype(np.float32)

            x_np[:, 2 * d:2 * d + 2] = C[i0:i1]

            future[s, i0:i1, 0:d] = x_np[:, 0:d]
            future[s, i0:i1, d:2 * d] = x_np[:, d:2 * d]

            X_t = to_tensor(x_np, device)

    return future.reshape(n_roll, H, W, 2 * d)
# ==================== AE ROLLOUT FROM START (VERIFICATION) ====================
@torch.no_grad()
def predict_rollout_from_start_ae_predictor(
    td,
    encoder,
    decoder,
    device,
    *,
    steps: int,
    escape_r: float,
    batch_size: int = 50000,
) -> np.ndarray:


    if td.X_grid is None:
        raise ValueError("AE ROLLOUT-FROM-START NEEDS td.X_grid")

    feat_dim = int(td.X_grid.shape[-1])
    d = int((feat_dim - 2) // 2)
    H = int(td.X_grid.shape[1])
    W = int(td.X_grid.shape[2])
    r = float(escape_r)

    n_roll = max(int(steps) - 1, 0)

    X_1 = td.X_grid[0].reshape(-1, feat_dim).astype(np.float32)
    C = X_1[:, 2 * d:2 * d + 2].copy().astype(np.float32)

    P = int(X_1.shape[0])
    future = np.zeros((n_roll, P, 2 * d), dtype=np.float32)

    for i0 in range(0, P, int(batch_size)):
        i1 = min(P, i0 + int(batch_size))

        X_t = to_tensor(X_1[i0:i1], device)
        C_t = to_tensor(C[i0:i1], device)

        for s in range(n_roll):
            X_t = decoder(encoder(X_t))
            X_t[:, 2 * d] = C_t[:, 0]
            X_t[:, 2 * d + 1] = C_t[:, 1]

            x_np = X_t.detach().cpu().numpy().astype(np.float32)
            zr = x_np[:, 0:d]
            zi = x_np[:, d:2 * d]
            mag = np.sqrt(np.maximum(zr * zr + zi * zi, 1e-30)).astype(np.float32)
            bad = (~np.isfinite(mag)) | (mag > r)

            if np.any(bad):
                safe = np.where((mag > 0.0) & np.isfinite(mag), mag, 1.0).astype(np.float32)
                scale = (r / safe).astype(np.float32)
                x_np[:, 0:d] = np.where(bad, zr * scale, zr).astype(np.float32)
                x_np[:, d:2 * d] = np.where(bad, zi * scale, zi).astype(np.float32)

            x_np[:, 2 * d:2 * d + 2] = C[i0:i1]

            future[s, i0:i1, 0:d] = x_np[:, 0:d]
            future[s, i0:i1, d:2 * d] = x_np[:, d:2 * d]

            X_t = to_tensor(x_np, device)

    return future.reshape(n_roll, H, W, 2 * d)
@torch.no_grad()
def teacher_forced_escape_iters_ae_predictor(td, encoder, decoder, device, escape_r: float, batch_size: int = 50000) -> np.ndarray:

    if td.X_grid is None:
        raise ValueError("AE TEACHER FORCED FRACTAL NEEDS td.X_grid")

    feat_dim = int(td.X_grid.shape[-1])
    d = int((feat_dim - 2) // 2)
    T = int(td.X_grid.shape[0])
    H = int(td.X_grid.shape[1])
    W = int(td.X_grid.shape[2])
    r2 = float(escape_r) * float(escape_r)

    P = H * W
    iters = np.full((P,), int(T), dtype=np.int32)

    for t in range(0, T - 1):
        X_t = td.X_grid[t].reshape(-1, feat_dim).astype(np.float32)
        C = X_t[:, 2 * d:2 * d + 2]
        esc_pred = np.zeros((P,), dtype=bool)

        for i0 in range(0, P, int(batch_size)):
            i1 = min(P, i0 + int(batch_size))
            xb = to_tensor(X_t[i0:i1], device)
            xk1 = decoder(encoder(xb))
            xk1[:, 2 * d] = to_tensor(C[i0:i1, 0], device)
            xk1[:, 2 * d + 1] = to_tensor(C[i0:i1, 1], device)

            x_np = xk1.detach().cpu().numpy().astype(np.float32)
            zr = x_np[:, 0:d]
            zi = x_np[:, d:2 * d]
            comp_mag2 = zr * zr + zi * zi
            max_mag2 = np.max(np.where(np.isfinite(comp_mag2), comp_mag2, np.inf), axis=1)
            esc_pred[i0:i1] = (~np.isfinite(max_mag2)) | (max_mag2 >= r2)

        new_esc = esc_pred & (iters == int(T))
        iters[new_esc] = int(t + 2)

    return iters.reshape(H, W)


#DMD ONLY HELPERS
def fit_full_state_dmd_streamed(X1, X2, *, device: torch.device, ridge: float | None = None):
    if ridge is None:
        ridge = D.DMD_RIDGE
    X1_list = [X1] if isinstance(X1, np.ndarray) else list(X1)
    X2_list = [X2] if isinstance(X2, np.ndarray) else list(X2)

    if len(X1_list) != len(X2_list):
        raise ValueError("DMD ONLY X1 AND X2 LISTS MUST HAVE SAME LENGTH")

    feat_dim = int(X1_list[0].shape[1])
    G = np.zeros((feat_dim, feat_dim), dtype=np.float64)
    H = np.zeros((feat_dim, feat_dim), dtype=np.float64)

    for a, b in zip(X1_list, X2_list):
        A = np.asarray(a, dtype=np.float64)
        B = np.asarray(b, dtype=np.float64)
        G += A.T @ A
        H += A.T @ B

    return fit_dmd_from_latent_covariances(G, H, device=device, ridge=ridge)


@torch.no_grad()
def dmd_only_one_step_metrics(dmd, X1: np.ndarray, X2: np.ndarray, device) -> dict:
    x1 = to_tensor(X1.astype(np.float32), device)
    x2_hat = dmd.predict(x1, steps=1)[-1]
    pred = x2_hat.detach().cpu().numpy().astype(np.float32)

    return {
        "dmd_only_mse": _mse(pred, X2.astype(np.float32)),
        "dmd_only_rel_l2": _rel_l2(pred, X2.astype(np.float32)),
        "dmd_only_fit": float(1.0 - _rel_l2(pred, X2.astype(np.float32))),
    }


@torch.no_grad()
def predict_next_snapshot_dmd_only(td, dmd, device, *, steps: int, escape_r: float, batch_size: int = 50000) -> np.ndarray:

    if td.X_grid is None:
        raise ValueError("DMD ONLY PREDICT NEXT NEEDS td.X_grid")

    feat_dim = int(td.X_grid.shape[-1])
    d = int((feat_dim - 2) // 2)
    H = int(td.X_grid.shape[1])
    W = int(td.X_grid.shape[2])
    r = float(escape_r)

    X_T = td.X_grid[-1].reshape(-1, feat_dim).astype(np.float32)
    C = X_T[:, 2 * d:2 * d + 2].copy().astype(np.float32)
    X_out = np.zeros((X_T.shape[0], 2 * d), dtype=np.float32)
    n_roll = max(int(steps), 0)

    for i0 in range(0, X_T.shape[0], int(batch_size)):
        i1 = min(X_T.shape[0], i0 + int(batch_size))
        X_t = to_tensor(X_T[i0:i1], device)
        C_t = to_tensor(C[i0:i1], device)

        for _ in range(n_roll):
            X_t = dmd.predict(X_t, steps=1)[-1]
            X_t[:, 2 * d] = C_t[:, 0]
            X_t[:, 2 * d + 1] = C_t[:, 1]

            x_np = X_t.detach().cpu().numpy().astype(np.float32)
            zr = x_np[:, 0:d]
            zi = x_np[:, d:2 * d]
            mag = np.sqrt(np.maximum(zr * zr + zi * zi, 1e-30)).astype(np.float32)
            bad = (~np.isfinite(mag)) | (mag > r)
            if np.any(bad):
                safe = np.where((mag > 0.0) & np.isfinite(mag), mag, 1.0).astype(np.float32)
                scale = (r / safe).astype(np.float32)
                x_np[:, 0:d] = np.where(bad, zr * scale, zr).astype(np.float32)
                x_np[:, d:2 * d] = np.where(bad, zi * scale, zi).astype(np.float32)
            X_t = to_tensor(x_np, device)

        x_final = X_t.detach().cpu().numpy().astype(np.float32)
        X_out[i0:i1, 0:d] = x_final[:, 0:d]
        X_out[i0:i1, d:2 * d] = x_final[:, d:2 * d]

    return X_out.reshape(H, W, 2 * d)


#DATA BUILDERS
def _build_single_td(train_clip_r: float | None = None):



    clip_r = float(D.DYNAMICS_CLAMP_R if train_clip_r is None else train_clip_r)

    return build_matrix_c_grid_training_data(
        data_dir=D.A_DATA_DIR,
        source=D.SINGLE_MATRIX_SOURCE,
        index=D.SINGLE_MATRIX_INDEX,
        c_re_min=D.C_RE_MIN,
        c_re_max=D.C_RE_MAX,
        c_im_min=D.C_IM_MIN,
        c_im_max=D.C_IM_MAX,
        c_re_n=D.SINGLE_MATRIX_C_RE_N,
        c_im_n=D.SINGLE_MATRIX_C_IM_N,
        max_iters=D.TRAIN_MAX_ITERS,
        escape_r=clip_r,
        classify_r=D.ESCAPE_R,
        filter_escaped=D.FILTER_ESCAPED_FOR_TRAINING,
    )


def _build_multi_train_test():
    A_all = load_all_A_matrices(D.A_DATA_DIR, source=D.MULTI_MATRIX_SOURCE)
    total = int(A_all.shape[0])

    train_idx, test_idx = split_explicit_matrix_indices(
        total_count=total,
        train_count=D.MULTI_MATRIX_TRAIN_COUNT,
        test_count=D.MULTI_MATRIX_TEST_COUNT,
        seed=D.MULTI_MATRIX_SPLIT_SEED,
    )

    td_train_list = build_matrix_c_grid_training_data_many_matrices(
        data_dir=D.A_DATA_DIR, source=D.MULTI_MATRIX_SOURCE, indices=train_idx,
        c_re_min=D.C_RE_MIN, c_re_max=D.C_RE_MAX, c_im_min=D.C_IM_MIN, c_im_max=D.C_IM_MAX,
        c_re_n=D.MULTI_MATRIX_C_RE_N, c_im_n=D.MULTI_MATRIX_C_IM_N,
        max_iters=D.TRAIN_MAX_ITERS, escape_r=D.DYNAMICS_CLAMP_R,
        classify_r=D.ESCAPE_R, filter_escaped=D.FILTER_ESCAPED_FOR_TRAINING,
    )

    td_test_list = build_matrix_c_grid_training_data_many_matrices(
        data_dir=D.A_DATA_DIR, source=D.MULTI_MATRIX_SOURCE, indices=test_idx,
        c_re_min=D.C_RE_MIN, c_re_max=D.C_RE_MAX, c_im_min=D.C_IM_MIN, c_im_max=D.C_IM_MAX,
        c_re_n=D.MULTI_MATRIX_C_RE_N, c_im_n=D.MULTI_MATRIX_C_IM_N,
        max_iters=D.TRAIN_MAX_ITERS, escape_r=D.DYNAMICS_CLAMP_R,
        classify_r=D.ESCAPE_R, filter_escaped=D.FILTER_ESCAPED_FOR_TRAINING,
    )

    return train_idx, test_idx, td_train_list, td_test_list


# AE-ONLY-SINGLE
def run_ae_only_single(device: torch.device) -> None:
    print("\n================ AE ONLY / SINGLE MATRIX ================\n")
    dirs = make_out_dirs("ae-only-single")

    A = load_one_A_matrix(D.A_DATA_DIR, source=D.SINGLE_MATRIX_SOURCE, index=D.SINGLE_MATRIX_INDEX)
    td = _build_single_td(float(getattr(D, "AE_PRED_TRAIN_CLIP_R", D.ESCAPE_R)))

    save_training_npz(dirs["td"] / "training_single_matrix.npz", td)
    save_ground_truth_escape_iters(td, D.ESCAPE_R, dirs["td"] / "gt_escape_iters.png")
    save_ground_truth_final_mask(td, D.ESCAPE_R, dirs["td"] / "gt_final_mask.png", scale=D.IMAGE_SCALE)

    train_kwargs = dict(
        latent_dim=int(getattr(D, "AE_PRED_LATENT_DIM", D.LATENT_DIM)),
        epochs=int(getattr(D, "AE_PRED_EPOCHS", getattr(D, "AE_ONLY_EPOCHS", D.AE_EPOCHS))),
        batch_size=int(getattr(D, "AE_ONLY_BATCH_SIZE", D.AE_BATCH_SIZE)),
        lr=float(getattr(D, "AE_ONLY_LR", D.AE_LR)),
        device=device,
    )
    if bool(getattr(D, "AE_USE_QUADRATIC_PREDICTOR", False)):
        state_dim = int(td.meta["state_dim"])
        q_enc, q_dec = make_quadratic_predictor_pair(
            state_dim, rank=getattr(D, "AE_QUADRATIC_RANK", None),
        )
        enc, dec, losses = train_autoencoder_predict_next_only(
            td.X1, td.X2, encoder=q_enc, decoder=q_dec, **train_kwargs,
        )

        plot_learned_vs_true_matrix_spectrum(
            A, q_enc.A.weight, dirs["res"] / "quadratic_A_spectrum.png",
        )
    else:
        enc, dec, losses = train_autoencoder_predict_next_only(td.X1, td.X2, **train_kwargs)

    save_loss_curve(losses, dirs["res"] / "loss_curve.png", "AE Predictor Single Matrix Loss")


    k = int(D.PREDICT_EXTRA_STEPS)
    future_pred = predict_future_snapshots_ae_predictor(td, enc, dec, device, steps=k, escape_r=D.DYNAMICS_CLAMP_R)

    m = ae_predictor_one_step_metrics(enc, dec, td.X1, td.X2, device)
    print_metric_block("AE PREDICTOR SINGLE (ONE-STEP)", m)

    all_metrics = {**m, "predict_extra_steps": float(k)}

    for s in range(k):
        step = s + 1

        Z_pred = future_pred[s]
        Z_true = iterate_true_next_snapshot(td, A, steps=step, escape_r=D.DYNAMICS_CLAMP_R)

        save_final_snapshot_image(
            Z_pred,
            escape_r=D.ESCAPE_R,
            out_png=dirs["res"] / f"pred_xn_plus_{step:03d}_mask.png",
            mode="mask",
        )

        save_final_snapshot_image(
            Z_pred,
            escape_r=D.ESCAPE_R,
            out_png=dirs["res"] / f"pred_xn_plus_{step:03d}_mag.png",
            mode="mag",
        )

        save_final_snapshot_image(
            Z_true,
            escape_r=D.ESCAPE_R,
            out_png=dirs["res"] / f"true_xn_plus_{step:03d}_mask.png",
            mode="mask",
        )

        save_final_snapshot_image(
            Z_true,
            escape_r=D.ESCAPE_R,
            out_png=dirs["res"] / f"true_xn_plus_{step:03d}_mag.png",
            mode="mag",
        )

        pred_m = next_step_prediction_metrics(Z_pred, Z_true)
        print_metric_block(f"AE PREDICTOR SINGLE xN+{step}", pred_m)

        all_metrics[f"pred_mse_xn_plus_{step:03d}"] = float(pred_m["pred_mse"])
        all_metrics[f"pred_rel_l2_xn_plus_{step:03d}"] = float(pred_m["pred_rel_l2"])
        all_metrics[f"pred_fit_xn_plus_{step:03d}"] = float(pred_m["pred_fit"])


    maxit = int(D.TRAIN_MAX_ITERS)
    rollout = predict_rollout_from_start_ae_predictor(
        td, enc, dec, device, steps=maxit, escape_r=D.DYNAMICS_CLAMP_R,
    )


    d_state = int((int(td.X_grid.shape[-1]) - 2) // 2)

    Z_pred_final = rollout[-1]
    Z_true_final = td.X_grid[-1][..., :2 * d_state]
    save_final_snapshot_image(Z_pred_final, escape_r=D.ESCAPE_R,
                              out_png=dirs["res"] / "rollout_from_start_final_mask.png",
                              mode="mask")
    save_final_snapshot_image(Z_pred_final, escape_r=D.ESCAPE_R,
                              out_png=dirs["res"] / "rollout_from_start_final_mag.png", mode="mag")

    final_m = next_step_prediction_metrics(Z_pred_final, Z_true_final)
    print_metric_block(f"AE ROLLOUT FROM x1 -> x{maxit} (MACRO, FULL GRID)", final_m)
    all_metrics.update({f"rollout_final_{key}": val for key, val in final_m.items()})

    alive_grid = td.meta.get("alive_mask_grid", None)
    if alive_grid is not None and bool(np.any(alive_grid)):
        final_m_alive = next_step_prediction_metrics(Z_pred_final[alive_grid], Z_true_final[alive_grid])
        print_metric_block(
            f"AE ROLLOUT FROM x1 -> x{maxit} (MACRO, ALIVE-ONLY, "
            f"{int(np.count_nonzero(alive_grid))}/{alive_grid.size} px)",
            final_m_alive,
        )
        all_metrics.update({f"rollout_final_alive_{key}": val for key, val in final_m_alive.items()})


    n_check = min(int(getattr(D, "PREDICT_ROLLOUT_CHECK_STEPS", 10)), rollout.shape[0])
    rollout_rel_l2: list[float] = []
    rollout_rel_l2_alive: list[float] = []

    for s in range(n_check):
        true_iter = s + 2
        Z_pred_s = rollout[s]
        Z_true_s = td.X_grid[s + 1][..., :2 * d_state]

        m_s = next_step_prediction_metrics(Z_pred_s, Z_true_s)
        print_metric_block(f"AE ROLLOUT FROM x1, {s + 1} STEP(S) IN (x{true_iter}, FULL GRID)", m_s)

        all_metrics[f"rollout_rel_l2_step_{s + 1:03d}"] = float(m_s["pred_rel_l2"])
        rollout_rel_l2.append(float(m_s["pred_rel_l2"]))

        if alive_grid is not None and bool(np.any(alive_grid)):
            m_s_alive = next_step_prediction_metrics(Z_pred_s[alive_grid], Z_true_s[alive_grid])
            print_metric_block(f"AE ROLLOUT FROM x1, {s + 1} STEP(S) IN (x{true_iter}, ALIVE-ONLY)", m_s_alive)
            all_metrics[f"rollout_rel_l2_alive_step_{s + 1:03d}"] = float(m_s_alive["pred_rel_l2"])
            rollout_rel_l2_alive.append(float(m_s_alive["pred_rel_l2"]))

    save_loss_curve(
        rollout_rel_l2, dirs["res"] / "rollout_rel_l2_vs_step.png",
        "AE Rollout Relative L2 Error vs Steps Beyond x1 (Full Grid)",
        xlabel="Steps beyond x1", ylabel="Relative L2 error",
    )
    if rollout_rel_l2_alive:
        save_loss_curve(
            rollout_rel_l2_alive, dirs["res"] / "rollout_rel_l2_vs_step_alive.png",
            f"AE Rollout Relative L2 Error vs Steps Beyond x1 (Alive-Only, {int(np.count_nonzero(alive_grid))}/{alive_grid.size} px)",
            xlabel="Steps beyond x1", ylabel="Relative L2 error",
        )

    write_metrics_txt(dirs["res"] / "metrics.txt", all_metrics)


#AE-ONLY-MULTI
def run_ae_only_multi(device: torch.device) -> None:
    print("\n================ AE ONLY / MULTIPLE MATRICES ================\n")
    dirs = make_out_dirs("ae-only-multi")

    A_all = load_all_A_matrices(D.A_DATA_DIR, source=D.MULTI_MATRIX_SOURCE)
    train_idx, test_idx, td_train_list, td_test_list = _build_multi_train_test()
    print("TRAIN COUNT:", int(train_idx.size), "TEST COUNT:", int(test_idx.size))

    save_ground_truth_final_mask(td_train_list[0], D.ESCAPE_R, dirs["td"] / "train_example_gt_final_mask.png", scale=D.IMAGE_SCALE)
    save_training_npz(dirs["td"] / "training_train_example.npz", td_train_list[0])

    train_kwargs = dict(
        latent_dim=int(getattr(D, "AE_PRED_LATENT_DIM", D.LATENT_DIM)),
        epochs=int(getattr(D, "AE_PRED_EPOCHS", getattr(D, "AE_ONLY_EPOCHS", D.AE_EPOCHS))),
        batch_size=int(getattr(D, "AE_ONLY_BATCH_SIZE", D.AE_BATCH_SIZE)),
        lr=float(getattr(D, "AE_ONLY_LR", D.AE_LR)),
        device=device,
    )
    if bool(getattr(D, "AE_USE_QUADRATIC_PREDICTOR", False)):

        state_dim = int(td_train_list[0].meta["state_dim"])
        q_enc, q_dec = make_quadratic_predictor_pair(
            state_dim, rank=getattr(D, "AE_QUADRATIC_RANK", None),
        )
        enc, dec, losses = train_autoencoder_predict_next_only(
            [td.X1 for td in td_train_list], [td.X2 for td in td_train_list],
            encoder=q_enc, decoder=q_dec, **train_kwargs,
        )
    else:
        enc, dec, losses = train_autoencoder_predict_next_only(
            [td.X1 for td in td_train_list], [td.X2 for td in td_train_list], **train_kwargs,
        )
    save_loss_curve(losses, dirs["res"] / "loss_curve.png", "AE Predictor Multiple Matrices Loss")

    k = int(D.PREDICT_EXTRA_STEPS)
    metric_list = []
    pred_metric_list = []

    maxit = int(D.TRAIN_MAX_ITERS)
    n_check = min(int(getattr(D, "PREDICT_ROLLOUT_CHECK_STEPS", 10)), max(maxit - 1, 0))
    rollout_final_metrics_all = []
    rollout_final_alive_metrics_all = []
    rollout_step_curves_all = []
    rollout_step_curves_alive_all = []

    for j, td_test in enumerate(td_test_list):
        A_test = A_all[int(test_idx[j])]
        alive_grid = td_test.meta.get("alive_mask_grid", None)
        mm = ae_predictor_one_step_metrics(enc, dec, td_test.X1, td_test.X2, device)

        Z_pred = predict_next_snapshot_ae_predictor(td_test, enc, dec, device, steps=k, escape_r=D.DYNAMICS_CLAMP_R)
        Z_true_next = iterate_true_next_snapshot(td_test, A_test, steps=k, escape_r=D.DYNAMICS_CLAMP_R)
        pred_m = next_step_prediction_metrics(Z_pred, Z_true_next)

        metric_list.append(mm)
        pred_metric_list.append(pred_m)
        print_metric_block(f"AE PREDICTOR TEST MATRIX {int(test_idx[j])} (ONE-STEP)", mm)
        print_metric_block(f"AE PREDICTOR TEST MATRIX {int(test_idx[j])} PREDICT (+{k})", pred_m)


        rollout = predict_rollout_from_start_ae_predictor(
            td_test, enc, dec, device, steps=maxit, escape_r=D.DYNAMICS_CLAMP_R,
        )
        d_state = int((int(td_test.X_grid.shape[-1]) - 2) // 2)

        Z_pred_final = rollout[-1]
        Z_true_final = td_test.X_grid[-1][..., :2 * d_state]
        final_m = next_step_prediction_metrics(Z_pred_final, Z_true_final)
        rollout_final_metrics_all.append(final_m)
        print_metric_block(f"AE PREDICTOR TEST MATRIX {int(test_idx[j])} ROLLOUT x1 -> x{maxit} (MACRO, FULL GRID)", final_m)

        if alive_grid is not None and bool(np.any(alive_grid)):
            final_m_alive = next_step_prediction_metrics(Z_pred_final[alive_grid], Z_true_final[alive_grid])
            rollout_final_alive_metrics_all.append(final_m_alive)
            print_metric_block(
                f"AE PREDICTOR TEST MATRIX {int(test_idx[j])} ROLLOUT x1 -> x{maxit} (MACRO, ALIVE-ONLY, "
                f"{int(np.count_nonzero(alive_grid))}/{alive_grid.size} px)", final_m_alive,
            )

        step_curve = []
        step_curve_alive = []
        for s in range(n_check):
            Z_pred_s = rollout[s]
            Z_true_s = td_test.X_grid[s + 1][..., :2 * d_state]
            m_s = next_step_prediction_metrics(Z_pred_s, Z_true_s)
            step_curve.append(float(m_s["pred_rel_l2"]))

            if alive_grid is not None and bool(np.any(alive_grid)):
                m_s_alive = next_step_prediction_metrics(Z_pred_s[alive_grid], Z_true_s[alive_grid])
                step_curve_alive.append(float(m_s_alive["pred_rel_l2"]))

        rollout_step_curves_all.append(step_curve)
        if step_curve_alive:
            rollout_step_curves_alive_all.append(step_curve_alive)

        if j == 0:
            save_ground_truth_final_mask(td_test, D.ESCAPE_R, dirs["res"] / "test_gt_final_mask.png", scale=D.IMAGE_SCALE)
            save_ground_truth_escape_iters(td_test, D.ESCAPE_R, dirs["res"] / "test_gt_escape_iters.png")

            Z_recon = reconstruct_final_snapshot_ae_only(td_test, enc, dec, device)
            save_final_snapshot_image(Z_recon, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "test_recon_final_mask.png",
                                       mode="mask", alive_mask=alive_grid)

            save_final_snapshot_image(Z_pred, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "test_pred_final_mask.png",
                                       mode="mask", alive_mask=alive_grid)
            save_final_snapshot_image(Z_true_next, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "test_true_next_final_mask.png", mode="mask")

            iters_test = teacher_forced_escape_iters_ae_predictor(td_test, enc, dec, device, D.ESCAPE_R)
            save_escape_image(iters_test, max_iters=int(td_test.X_grid.shape[0]), out_png=dirs["res"] / "test_pred_escape_iters.png",
                               alive_mask=alive_grid)

            save_final_snapshot_image(Z_pred_final, escape_r=D.ESCAPE_R,
                                       out_png=dirs["res"] / "test_rollout_from_start_final_mask.png",
                                       mode="mask", alive_mask=alive_grid)
            save_final_snapshot_image(Z_pred_final, escape_r=D.ESCAPE_R,
                                       out_png=dirs["res"] / "test_rollout_from_start_final_mag.png",
                                       mode="mag", alive_mask=alive_grid)

    mean_m = mean_metric_dict(metric_list)
    mean_pred = mean_metric_dict(pred_metric_list)
    print_metric_block("AE PREDICTOR MEAN TEST (ONE-STEP)", mean_m)
    print_metric_block(f"AE PREDICTOR MEAN TEST PREDICT (+{k})", mean_pred)


    mean_rollout_final = mean_metric_dict(rollout_final_metrics_all)
    print_metric_block(f"AE PREDICTOR MEAN TEST ROLLOUT x1 -> x{maxit} (MACRO, FULL GRID)", mean_rollout_final)
    mean_rollout_final_alive = {}
    if rollout_final_alive_metrics_all:
        mean_rollout_final_alive_raw = mean_metric_dict(rollout_final_alive_metrics_all)
        print_metric_block(f"AE PREDICTOR MEAN TEST ROLLOUT x1 -> x{maxit} (MACRO, ALIVE-ONLY)", mean_rollout_final_alive_raw)
        mean_rollout_final_alive = {f"rollout_final_alive_{key}": val for key, val in mean_rollout_final_alive_raw.items()}

    rollout_step_metrics: dict[str, float] = {}
    mean_step_curve: list[float] = []
    if rollout_step_curves_all:
        mean_step_curve = np.mean(np.asarray(rollout_step_curves_all, dtype=np.float64), axis=0).tolist()
        for s, val in enumerate(mean_step_curve):
            rollout_step_metrics[f"rollout_rel_l2_step_{s + 1:03d}"] = float(val)

    mean_step_curve_alive: list[float] = []
    if rollout_step_curves_alive_all:
        mean_step_curve_alive = np.mean(np.asarray(rollout_step_curves_alive_all, dtype=np.float64), axis=0).tolist()
        for s, val in enumerate(mean_step_curve_alive):
            rollout_step_metrics[f"rollout_rel_l2_alive_step_{s + 1:03d}"] = float(val)

    if mean_step_curve:
        save_loss_curve(
            mean_step_curve, dirs["res"] / "rollout_rel_l2_vs_step.png",
            f"AE Rollout Relative L2 Error vs Steps Beyond x1 (Mean Over {len(rollout_step_curves_all)} Test Matrices, Full Grid)",
            xlabel="Steps beyond x1", ylabel="Relative L2 error", log_scale=False,
        )
    if mean_step_curve_alive:
        save_loss_curve(
            mean_step_curve_alive, dirs["res"] / "rollout_rel_l2_vs_step_alive.png",
            f"AE Rollout Relative L2 Error vs Steps Beyond x1 (Mean Over {len(rollout_step_curves_alive_all)} Test Matrices, Alive-Only)",
            xlabel="Steps beyond x1", ylabel="Relative L2 error", log_scale=False,
        )

    write_metrics_txt(dirs["res"] / "mean_test_metrics.txt", {
        **mean_m, **mean_pred, "predict_extra_steps": float(k),
        **{f"rollout_final_{key}": val for key, val in mean_rollout_final.items()},
        **mean_rollout_final_alive,
        **rollout_step_metrics,
    })


# DMD-ONLY-SINGLE
def run_dmd_only_single(device: torch.device) -> None:
    print("\n================ DMD ONLY / SINGLE MATRIX ================\n")
    dirs = make_out_dirs("dmd-only-single")

    A = load_one_A_matrix(D.A_DATA_DIR, source=D.SINGLE_MATRIX_SOURCE, index=D.SINGLE_MATRIX_INDEX)
    td = _build_single_td()

    save_training_npz(dirs["td"] / "training_single_matrix.npz", td)
    save_ground_truth_escape_iters(td, D.ESCAPE_R, dirs["td"] / "gt_escape_iters.png")
    save_ground_truth_final_mask(td, D.ESCAPE_R, dirs["td"] / "gt_final_mask.png", scale=D.IMAGE_SCALE)

    dmd = fit_full_state_dmd_streamed(td.X1, td.X2, device=device)
    rho = float(np.max(np.abs(np.linalg.eigvals(dmd.A.detach().cpu().numpy()))))
    print("DMD ONLY SINGLE SPECTRAL RADIUS:", rho)

    k = int(D.PREDICT_EXTRA_STEPS)
    Z_pred = predict_next_snapshot_dmd_only(td, dmd, device, steps=k, escape_r=D.DYNAMICS_CLAMP_R)
    Z_true_next = iterate_true_next_snapshot(td, A, steps=k, escape_r=D.DYNAMICS_CLAMP_R)

    _save_final_mask_95(Z_pred, D.ESCAPE_R, dirs["res"] / "pred_final_mask.png", scale=D.IMAGE_SCALE)
    save_final_snapshot_image(Z_pred, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "pred_final_snapshot_mag.png", mode="mag")
    _save_final_mask_95(Z_true_next, D.ESCAPE_R, dirs["res"] / "true_next_final_mask.png", scale=D.IMAGE_SCALE)

    m = dmd_only_one_step_metrics(dmd, td.X1, td.X2, device)
    pred_m = next_step_prediction_metrics(Z_pred, Z_true_next)
    m["dmd_only_spectral_radius"] = rho
    print_metric_block("DMD ONLY SINGLE (ONE-STEP)", m)
    print_metric_block(f"DMD ONLY SINGLE PREDICT (+{k})", pred_m)
    write_metrics_txt(dirs["res"] / "metrics.txt", {**m, **pred_m, "predict_extra_steps": float(k)})


#DMD-ONLY-MULTI
def run_dmd_only_multi(device: torch.device) -> None:
    print("\n================ DMD ONLY / MULTIPLE MATRICES ================\n")
    dirs = make_out_dirs("dmd-only-multi")

    A_all = load_all_A_matrices(D.A_DATA_DIR, source=D.MULTI_MATRIX_SOURCE)
    train_idx, test_idx, td_train_list, td_test_list = _build_multi_train_test()
    print("TRAIN COUNT:", int(train_idx.size), "TEST COUNT:", int(test_idx.size))

    save_ground_truth_final_mask(td_train_list[0], D.ESCAPE_R, dirs["td"] / "train_example_gt_final_mask.png", scale=D.IMAGE_SCALE)
    save_training_npz(dirs["td"] / "training_train_example.npz", td_train_list[0])

    dmd = fit_full_state_dmd_streamed(
        [td.X1 for td in td_train_list], [td.X2 for td in td_train_list], device=device,
    )
    rho = float(np.max(np.abs(np.linalg.eigvals(dmd.A.detach().cpu().numpy()))))
    print("DMD ONLY MULTI SPECTRAL RADIUS:", rho)

    k = int(D.PREDICT_EXTRA_STEPS)
    metric_list = []
    pred_metric_list = []

    for j, td_test in enumerate(td_test_list):
        A_test = A_all[int(test_idx[j])]
        mm = dmd_only_one_step_metrics(dmd, td_test.X1, td_test.X2, device)
        mm["dmd_only_spectral_radius"] = rho

        Z_pred = predict_next_snapshot_dmd_only(td_test, dmd, device, steps=k, escape_r=D.DYNAMICS_CLAMP_R)
        Z_true_next = iterate_true_next_snapshot(td_test, A_test, steps=k, escape_r=D.DYNAMICS_CLAMP_R)
        pred_m = next_step_prediction_metrics(Z_pred, Z_true_next)

        metric_list.append(mm)
        pred_metric_list.append(pred_m)
        print_metric_block(f"DMD ONLY TEST MATRIX {int(test_idx[j])} (ONE-STEP)", mm)
        print_metric_block(f"DMD ONLY TEST MATRIX {int(test_idx[j])} PREDICT (+{k})", pred_m)

        if j == 0:
            save_ground_truth_final_mask(td_test, D.ESCAPE_R, dirs["res"] / "test_gt_final_mask.png", scale=D.IMAGE_SCALE)
            _save_final_mask_95(Z_pred, D.ESCAPE_R, dirs["res"] / "test_pred_final_mask.png", scale=D.IMAGE_SCALE)
            save_final_snapshot_image(Z_pred, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "test_pred_final_snapshot_mag.png", mode="mag")
            _save_final_mask_95(Z_true_next, D.ESCAPE_R, dirs["res"] / "test_true_next_final_mask.png", scale=D.IMAGE_SCALE)

    mean_m = mean_metric_dict(metric_list)
    mean_pred = mean_metric_dict(pred_metric_list)
    print_metric_block("DMD ONLY MEAN TEST (ONE-STEP)", mean_m)
    print_metric_block(f"DMD ONLY MEAN TEST PREDICT (+{k})", mean_pred)
    write_metrics_txt(dirs["res"] / "mean_test_metrics.txt", {**mean_m, **mean_pred, "predict_extra_steps": float(k)})