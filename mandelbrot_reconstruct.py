
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from utils import to_tensor

DEAD_PIXEL_TINT = (255, 205, 205)

#GRID BUILDE
def build_c_grid(
    *,
    c_re_min: float,
    c_re_max: float,
    c_im_min: float,
    c_im_max: float,
    grid_n: int,
) -> np.ndarray:
    xs = np.linspace(c_re_min, c_re_max, grid_n, dtype=np.float32)
    ys = np.linspace(c_im_min, c_im_max, grid_n, dtype=np.float32)
    C = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2).astype(np.float32)
    return C

#  CLAMP
def _cap_to_escape(zr: np.ndarray, zi: np.ndarray, escape_r: float):
    r = float(escape_r)
    mag = np.sqrt(zr * zr + zi * zi).astype(np.float32)
    bad = (~np.isfinite(mag)) | (mag > r)
    if not np.any(bad):
        return zr, zi
    mag_safe = np.where((mag > 0.0) & np.isfinite(mag), mag, 1.0).astype(np.float32)
    s = (r / mag_safe).astype(np.float32)
    zr2 = np.where(bad, zr * s, zr).astype(np.float32)
    zi2 = np.where(bad, zi * s, zi).astype(np.float32)
    return zr2, zi2

# RECONSTRUCT MANDELBROT ( i mean fractal but i got too lazy to replace the name of the module)
@torch.no_grad()
def reconstruct_mandelbrot(
    *,
    encoder,
    decoder,
    dmd,
    C: np.ndarray,
    grid_n: int,
    max_iters: int,
    escape_r: float,
    device: torch.device,
    state_dim: int,
    feat_dim: int,
) -> np.ndarray:
    P = int(C.shape[0])
    r2 = float(escape_r) * float(escape_r)

    X = np.zeros((P, feat_dim), dtype=np.float32)
    X[:, 0:state_dim] = C[:, 0:1].astype(np.float32)
    X[:, state_dim:2 * state_dim] = C[:, 1:2].astype(np.float32)
    X[:, 2 * state_dim] = C[:, 0].astype(np.float32)
    X[:, 2 * state_dim + 1] = C[:, 1].astype(np.float32)

    alive = np.ones((P,), dtype=bool)
    iters = np.zeros((P,), dtype=np.int32)


    zr0 = X[:, 0:state_dim]
    zi0 = X[:, state_dim:2 * state_dim]
    comp_mag2_0 = zr0 * zr0 + zi0 * zi0
    max_mag2_0 = np.max(comp_mag2_0, axis=1)
    esc0 = max_mag2_0 > r2
    iters[esc0] = 1
    alive[esc0] = False

    X_t = to_tensor(X, device)
    C_t = to_tensor(C.astype(np.float32), device)

    for k in range(2, int(max_iters) + 1):
        if not alive.any():
            break

        idx = np.where(alive)[0]
        xk = X_t[idx]

        zk = encoder(xk)
        zk1 = dmd.predict(zk, steps=1)[-1]
        xk1 = decoder(zk1)

        xk1[:, 2 * state_dim] = C_t[idx, 0]
        xk1[:, 2 * state_dim + 1] = C_t[idx, 1]

        x_cpu = xk1.detach().cpu().numpy().astype(np.float32)
        zr_all = x_cpu[:, 0:state_dim]
        zi_all = x_cpu[:, state_dim:2 * state_dim]

        comp_mag2 = zr_all * zr_all + zi_all * zi_all
        max_mag2 = np.max(comp_mag2, axis=1)
        esc = max_mag2 > r2

        bad = (~np.isfinite(zr_all)) | (~np.isfinite(zi_all))
        if np.any(bad):
            zr_all = np.where(np.isfinite(zr_all), zr_all, 0.0).astype(np.float32, copy=False)
            zi_all = np.where(np.isfinite(zi_all), zi_all, 0.0).astype(np.float32, copy=False)
            x_cpu[:, 0:state_dim] = zr_all
            x_cpu[:, state_dim:2 * state_dim] = zi_all

        xk1 = to_tensor(x_cpu, device)
        X_t[idx] = xk1

        escaped_idx = idx[esc]
        iters[escaped_idx] = k
        alive[escaped_idx] = False

    iters[alive] = int(max_iters)
    return iters.reshape(int(grid_n), int(grid_n))





def save_escape_image(
    escape_iters: np.ndarray,
    *,
    max_iters: int,
    out_png: str | Path,
    alive_mask: np.ndarray | None = None,
) -> str:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    esc = escape_iters.astype(np.float32)
    norm = esc / float(max_iters)
    img = (255.0 * norm).clip(0, 255).astype(np.uint8)
    img = 255 - img

    if alive_mask is None:
        im = Image.fromarray(img, mode="L")
    else:
        im = _tint_dead_pixels(img, alive_mask)

    im.save(out_png)
    return str(out_png)




def save_and_show_plot(
    escape_iters: np.ndarray,
    *,
    out_png: str | Path,
    show: bool,
) -> str:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.imshow(escape_iters, origin="lower")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    if show:
        plt.show()
    plt.close()
    return str(out_png)



@torch.no_grad()
def reconstruct_final_snapshot(
    *,
    encoder,
    decoder,
    dmd,
    C: np.ndarray,
    grid_n: int,
    steps: int,
    escape_r: float,
    device: torch.device,
    batch_size: int = 200000,
    state_dim: int,
    feat_dim: int,
) -> np.ndarray:
    P = int(C.shape[0])




    X = np.zeros((P, feat_dim), dtype=np.float32)
    X[:, 0:state_dim] = C[:, 0:1].astype(np.float32)
    X[:, state_dim:2 * state_dim] = C[:, 1:2].astype(np.float32)
    X[:, 2 * state_dim] = C[:, 0].astype(np.float32)
    X[:, 2 * state_dim + 1] = C[:, 1].astype(np.float32)


    n_roll = max(int(steps) - 1, 0)

    C_t_all = to_tensor(C.astype(np.float32), device)
    X_out = np.zeros((P, 2 * state_dim), dtype=np.float32)

    for i0 in range(0, P, int(batch_size)):
        i1 = min(P, i0 + int(batch_size))
        idx = slice(i0, i1)

        X_t = to_tensor(X[idx], device)
        C_t = C_t_all[idx]

        for _ in range(n_roll):
            zk = encoder(X_t)
            zk1 = dmd.predict(zk, steps=1)[-1]
            X_t = decoder(zk1)

            X_t[:, 2 * state_dim] = C_t[:, 0]
            X_t[:, 2 * state_dim + 1] = C_t[:, 1]

            x_cpu = X_t.detach().cpu().numpy().astype(np.float32)
            zr_all = x_cpu[:, 0:state_dim]
            zi_all = x_cpu[:, state_dim:2 * state_dim]

            comp_mag2 = zr_all * zr_all + zi_all * zi_all
            comp_mag = np.sqrt(np.maximum(comp_mag2, 1e-30)).astype(np.float32)
            bad = (~np.isfinite(comp_mag)) | (comp_mag > float(escape_r))

            if np.any(bad):
                safe_mag = np.where((comp_mag > 0.0) & np.isfinite(comp_mag), comp_mag, 1.0).astype(np.float32)
                scale = (float(escape_r) / safe_mag).astype(np.float32)
                zr_all = np.where(bad, zr_all * scale, zr_all).astype(np.float32)
                zi_all = np.where(bad, zi_all * scale, zi_all).astype(np.float32)
                x_cpu[:, 0:state_dim] = zr_all
                x_cpu[:, state_dim:2 * state_dim] = zi_all

            X_t = to_tensor(x_cpu, device)

        x_final = X_t.detach().cpu().numpy().astype(np.float32)
        X_out[idx, 0:state_dim] = x_final[:, 0:state_dim]
        X_out[idx, state_dim:2 * state_dim] = x_final[:, state_dim:2 * state_dim]

    return X_out.reshape(int(grid_n), int(grid_n), 2 * state_dim)

# FLAG OUT-OF-DOMAIN PIXELS
def _tint_dead_pixels(img_l: np.ndarray, alive_mask: np.ndarray) -> Image.Image:
    alive_mask = np.asarray(alive_mask, dtype=bool)
    if alive_mask.shape != img_l.shape:
        raise ValueError(f"alive_mask SHAPE {alive_mask.shape} != IMAGE SHAPE {img_l.shape}")

    rgb = np.stack([img_l, img_l, img_l], axis=-1).astype(np.uint8)
    tint = np.array(DEAD_PIXEL_TINT, dtype=np.uint8)
    rgb[~alive_mask] = tint
    return Image.fromarray(rgb, mode="RGB")

#SAVE FINAL SNAPSHOT IMAG
def save_final_snapshot_image(
    Z_final: np.ndarray,
    *,
    escape_r: float,
    out_png: str | Path,
    mode: str = "mag",
    alive_mask: np.ndarray | None = None,
) -> str:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    state_dim = int(Z_final.shape[-1] // 2)
    zr = Z_final[..., 0:state_dim].astype(np.float32, copy=False)
    zi = Z_final[..., state_dim:2 * state_dim].astype(np.float32, copy=False)
    comp_mag2 = zr * zr + zi * zi
    max_mag = np.sqrt(np.max(comp_mag2, axis=-1)).astype(np.float32)

    if str(mode).lower() == "mask":
        img = ((max_mag < float(escape_r)).astype(np.uint8) * 255)
    elif str(mode).lower() == "angle":
        ang = np.arctan2(zi[..., 0], zr[..., 0]).astype(np.float32)
        norm = (ang + np.pi) / (2.0 * np.pi)
        img = (255.0 * norm).clip(0, 255).astype(np.uint8)
    else:
        mag = np.minimum(max_mag, float(escape_r)).astype(np.float32)
        norm = mag / float(escape_r)
        img = (255.0 * norm).clip(0, 255).astype(np.uint8)

    if alive_mask is None:
        Image.fromarray(img, mode="L").save(out_png)
    else:
        _tint_dead_pixels(img, alive_mask).save(out_png)

    return str(out_png)