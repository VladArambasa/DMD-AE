
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from utils import to_tensor
import defines as D


def save_loss_curve(
    losses: list[float],
    out_png: str | Path,
    title: str,
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
    *,
    log_scale: bool | None = None,  
    extra_series: dict[str, list[float]] | None = None,
    primary_label: str | None = None,
) -> str:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    if log_scale is None:
        log_scale = bool(getattr(D, "LOSS_CURVES_LOG_SCALE", True))

    xs = np.arange(1, len(losses) + 1)
    ys = np.asarray(losses, dtype=np.float32)

    plt.figure()
    plot_fn = plt.semilogy if log_scale else plt.plot

    label = primary_label if extra_series else None
    if log_scale:
        ys_plot = np.where(ys > 0, ys, np.nan)
        plot_fn(xs, ys_plot, label=label)
    else:
        plot_fn(xs, ys, label=label)

    if extra_series:
        for name, series in extra_series.items():
            series = np.asarray(series, dtype=np.float32)
            xs_e = np.arange(1, len(series) + 1)
            if log_scale:
                series = np.where(series > 0, series, np.nan)
            plot_fn(xs_e, series, label=name)
        plt.legend()

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    return str(out_png)

@torch.no_grad()
def autoencoder_reconstruction_metrics(encoder, decoder, X: np.ndarray, device: torch.device, batch_size: int = 50000) -> dict: 
    N = int(X.shape[0])
    sse = 0.0
    sq_true = 0.0

    for i0 in range(0, N, int(batch_size)):
        i1 = min(N, i0 + int(batch_size))
        xb = to_tensor(X[i0:i1], device)
        xh_np = decoder(encoder(xb)).detach().cpu().numpy().astype(np.float32)
        err = xh_np - X[i0:i1].astype(np.float32)
        sse += float(np.sum(err * err))
        sq_true += float(np.sum(X[i0:i1].astype(np.float32) ** 2))

    mse = sse / max(N * X.shape[1], 1)
    rel_l2 = float(np.sqrt(sse) / max(np.sqrt(sq_true), 1e-12))
    fit = float(1.0 - rel_l2)
    return {"ae_mse": mse, "ae_rel_l2": rel_l2, "ae_fit": fit}


def _alive_row_mask(X: np.ndarray, escape_r: float) -> np.ndarray: 
    X = np.asarray(X, dtype=np.float32) 
    feat_dim = int(X.shape[1]) 
    d = int((feat_dim - 2) // 2) 

    zr = X[:, 0:d] 
    zi = X[:, d:2 * d] 

    mag2 = zr * zr + zi * zi 
    max_mag2 = np.max(np.where(np.isfinite(mag2), mag2, np.inf), axis=1) 

    finite = np.isfinite(zr).all(axis=1) & np.isfinite(zi).all(axis=1) 
    alive = finite & (max_mag2 < 0.999 * float(escape_r) * float(escape_r)) 

    return alive 


def _exact_trained_row_mask(td) -> np.ndarray: 
    meta = getattr(td, "meta", None) or {} 
    alive_grid = meta.get("alive_mask_grid", None) 
    T = meta.get("max_iters", None) 

    if alive_grid is None or T is None: 
        classify_r = float(meta.get("classify_r", 2.0)) 
        return _alive_row_mask(td.X, classify_r) 

    alive_flat = np.asarray(alive_grid, dtype=bool).reshape(-1) 
    P = int(alive_flat.shape[0]) 
    T = int(T) 

    if int(td.X.shape[0]) != T * P: 
        classify_r = float(meta.get("classify_r", 2.0)) 
        return _alive_row_mask(td.X, classify_r) 

    return np.tile(alive_flat, T) 


@torch.no_grad() 
def autoencoder_reconstruction_metrics_alive( 
    encoder, decoder, td, device: torch.device, batch_size: int = 50000,
) -> dict:
    X = np.asarray(td.X) 
    mask = _exact_trained_row_mask(td) 
    n_alive = int(np.count_nonzero(mask)) 

    if n_alive == 0: 
        return { 
            "ae_alive_mse": float("nan"), "ae_alive_rel_l2": float("nan"), "ae_alive_fit": float("nan"),
            "ae_alive_n": 0.0, "ae_alive_n_total": float(X.shape[0]),
        }

    base = autoencoder_reconstruction_metrics(encoder, decoder, X[mask], device, batch_size=batch_size) 
    return { 
        "ae_alive_mse": base["ae_mse"],
        "ae_alive_rel_l2": base["ae_rel_l2"],
        "ae_alive_fit": base["ae_fit"],
        "ae_alive_n": float(n_alive),
        "ae_alive_n_total": float(X.shape[0]),
    }

@torch.no_grad() 
def dmd_one_step_metrics(encoder, decoder, dmd, X1: np.ndarray, X2: np.ndarray, device: torch.device, batch_size: int = 50000) -> dict: 
    N = int(X1.shape[0]) 
    sse = 0.0 
    sq_true = 0.0 

    for i0 in range(0, N, int(batch_size)): 
        i1 = min(N, i0 + int(batch_size)) 
        x1b = to_tensor(X1[i0:i1], device) 
        z1 = encoder(x1b) 
        z2h = dmd.predict(z1, steps=1)[-1] 
        x2h_np = decoder(z2h).detach().cpu().numpy().astype(np.float32) 
        err = x2h_np - X2[i0:i1].astype(np.float32) 
        sse += float(np.sum(err * err)) 
        sq_true += float(np.sum(X2[i0:i1].astype(np.float32) ** 2)) 

    mse = sse / max(N * X1.shape[1], 1) 
    rel_l2 = float(np.sqrt(sse) / max(np.sqrt(sq_true), 1e-12)) 
    fit = float(1.0 - rel_l2) 
    return {"dmd_mse": mse, "dmd_rel_l2": rel_l2, "dmd_fit": fit} 

def save_ground_truth_final_mask(td, escape_r: float, out_png: str | Path, scale: int = 64) -> str: 
    out_png = Path(out_png) 
    out_png.parent.mkdir(parents=True, exist_ok=True) 

    feat_dim = int(td.X_grid.shape[-1]) 
    d = (feat_dim - 2) // 2 
    r2 = float(escape_r) * float(escape_r) 

    zr = td.X_grid[-1, :, :, 0:d].astype(np.float32, copy=False) 
    zi = td.X_grid[-1, :, :, d:2 * d].astype(np.float32, copy=False) 
    comp_mag2 = zr * zr + zi * zi 

    max_mag2 = np.max(comp_mag2, axis=-1) 
    mask = max_mag2 < r2 

    img = (mask.astype(np.uint8) * 255) 
    pil_img = Image.fromarray(img, mode="L") 
    pil_img = pil_img.resize( 
        (img.shape[1] * int(scale), img.shape[0] * int(scale)), 
        resample=Image.NEAREST, 
    )
    pil_img.save(out_png) 
    return str(out_png) 

@torch.no_grad() 
def reconstruct_true_final_snapshot(td, encoder, decoder, device: torch.device, batch_size: int = 50000) -> np.ndarray: 
    if td.X_grid is None: 
        raise ValueError("RECON NEEDS td.X_grid") 

    feat_dim = int(td.X_grid.shape[-1]) 
    d = (feat_dim - 2) // 2 
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

def iterate_true_next_snapshot(td, A: np.ndarray, *, steps: int, escape_r: float) -> np.ndarray: 
    if td.X_grid is None: 
        raise ValueError("TRUE NEXT NEEDS td.X_grid") 

    feat_dim = int(td.X_grid.shape[-1]) 
    d = (feat_dim - 2) // 2 
    H = int(td.X_grid.shape[1]) 
    W = int(td.X_grid.shape[2]) 
    r = float(escape_r) 
    r2 = r * r 

    A = np.asarray(A, dtype=np.float32) 
    X_T = td.X_grid[-1].reshape(-1, feat_dim).astype(np.float32) 
    z = (X_T[:, 0:d] + 1j * X_T[:, d:2 * d]).astype(np.complex64) 
    c = (X_T[:, 2 * d] + 1j * X_T[:, 2 * d + 1]).astype(np.complex64) 

    for _ in range(int(steps)): 
        Az = (z @ A.T).astype(np.complex64) 
        z = (Az * Az).astype(np.complex64) 
        z = (z + c[:, None]).astype(np.complex64) 

        mag2 = (z.real * z.real + z.imag * z.imag).astype(np.float32) 
        bad = (~np.isfinite(mag2)) | (mag2 > r2) 
        if np.any(bad): 
            mag = np.sqrt(np.maximum(mag2, 1e-30)).astype(np.float32) 
            scale = (r / mag).astype(np.float32) 
            z_real = np.where(bad, z.real * scale, z.real).astype(np.float32) 
            z_imag = np.where(bad, z.imag * scale, z.imag).astype(np.float32) 
            z = (z_real + 1j * z_imag).astype(np.complex64) 

    X_out = np.zeros((z.shape[0], 2 * d), dtype=np.float32) 
    X_out[:, 0:d] = z.real.astype(np.float32) 
    X_out[:, d:2 * d] = z.imag.astype(np.float32) 
    return X_out.reshape(H, W, 2 * d) 

@torch.no_grad() 
def predict_next_snapshot(td, encoder, decoder, dmd, device: torch.device, *, steps: int, escape_r: float, batch_size: int = 50000) -> np.ndarray: 
    if td.X_grid is None: 
        raise ValueError("PREDICT NEXT NEEDS td.X_grid") 

    feat_dim = int(td.X_grid.shape[-1]) 
    d = (feat_dim - 2) // 2 
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
            zk = encoder(X_t) 
            zk1 = dmd.predict(zk, steps=1)[-1] 
            X_t = decoder(zk1) 
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

def next_step_prediction_metrics(pred_grid: np.ndarray, true_grid: np.ndarray) -> dict: 
    p = pred_grid.astype(np.float32).reshape(-1) 
    t = true_grid.astype(np.float32).reshape(-1)
    err = p - t
    mse = float(np.mean(err * err))
    rel_l2 = float(np.linalg.norm(err) / max(np.linalg.norm(t), 1e-12))
    fit = float(1.0 - rel_l2)
    return {"pred_mse": mse, "pred_rel_l2": rel_l2, "pred_fit": fit}

@torch.no_grad() 
def predict_rollout_from_start_ae_dmd(
    td,
    encoder,
    decoder,
    dmd,
    device,
    *,
    steps: int,
    escape_r: float,
    batch_size: int = 50000,
) -> np.ndarray:
    if td.X_grid is None: 
        raise ValueError("AE+DMD ROLLOUT-FROM-START NEEDS td.X_grid") 

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
            zk = encoder(X_t) 
            zk1 = dmd.predict(zk, steps=1)[-1] 
            X_t = decoder(zk1) 
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
def teacher_forced_escape_iters(td, encoder, decoder, dmd, device: torch.device, *, escape_r: float, batch_size: int = 50000) -> np.ndarray: 
    if td.X_grid is None: 
        raise ValueError("TEACHER FORCED FRACTAL NEEDS td.X_grid") 

    feat_dim = int(td.X_grid.shape[-1]) 
    d = (feat_dim - 2) // 2 
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
            zk1 = dmd.predict(encoder(xb), steps=1)[-1] 
            xk1 = decoder(zk1) 
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

def save_ground_truth_escape_iters(td, escape_r: float, out_png: str | Path) -> str: 
    out_png = Path(out_png) 
    out_png.parent.mkdir(parents=True, exist_ok=True) 

    feat_dim = int(td.X_grid.shape[-1]) 
    d = (feat_dim - 2) // 2 
    T = int(td.X_grid.shape[0]) 
    r2 = float(escape_r) * float(escape_r) 

    zr = td.X_grid[:, :, :, 0:d] 
    zi = td.X_grid[:, :, :, d:2 * d]
    comp_mag2 = zr * zr + zi * zi
    max_mag2 = np.max(comp_mag2, axis=-1)
    escaped = max_mag2 >= r2

    first_escape = np.argmax(escaped, axis=0).astype(np.int32) + 1
    never_escaped = ~np.any(escaped, axis=0)
    first_escape[never_escaped] = T

    img = (255.0 * (1.0 - first_escape.astype(np.float32) / float(T))).clip(0, 255).astype(np.uint8) 
    Image.fromarray(img, mode="L").save(out_png)
    return str(out_png)