from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

import defines as D
from utils import to_tensor
from apply_dmd import fit_dmd_from_latent_covariances


def pick_device() -> torch.device:
    if D.USE_CUDA_IF_AVAILABLE and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_out_dirs(mode: str) -> dict:
    root = Path("out") / mode
    td = root / "training-data"
    res = root / "results"
    td.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)
    return {"root": root, "td": td, "res": res}


def fit_streamed_dmd_from_td_list(enc, td_list: list, device: torch.device, ridge: float | None = None):

    if ridge is None:
        ridge = D.DMD_RIDGE
    latent_dim = int(D.LATENT_DIM)
    G = np.zeros((latent_dim, latent_dim), dtype=np.float64)
    H = np.zeros((latent_dim, latent_dim), dtype=np.float64)

    with torch.no_grad():
        for td in td_list:
            Z1 = enc(to_tensor(td.X1, device)).detach().cpu().numpy().astype(np.float64)
            Z2 = enc(to_tensor(td.X2, device)).detach().cpu().numpy().astype(np.float64)
            G += Z1.T @ Z1
            H += Z1.T @ Z2

    return fit_dmd_from_latent_covariances(G, H, device=device, ridge=ridge)


def print_metric_block(name: str, metrics: dict) -> None:
    print(f"[{name}]")
    for k, v in metrics.items():
        print(f"  {k}: {v:.8e}")


def mean_metric_dict(metric_list: list) -> dict:
    keys = metric_list[0].keys()
    out = {}
    for k in keys:
        out[k] = float(np.mean([m[k] for m in metric_list]))
    return out


def write_metrics_txt(path, metrics: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.10e}\n")


def debug_final_state_stats(name: str, Z_final: np.ndarray, escape_r: float) -> None:
    d = Z_final.shape[-1] // 2
    zr = Z_final[..., 0:d].astype(np.float32)
    zi = Z_final[..., d:2 * d].astype(np.float32)
    mag2 = zr * zr + zi * zi
    max_mag = np.sqrt(np.max(mag2, axis=-1))

    print(f"\n[{name}] FINAL PRED STATS")
    print("min max_mag:", float(np.min(max_mag)))
    print("mean max_mag:", float(np.mean(max_mag)))
    print("max max_mag:", float(np.max(max_mag)))
    print("escape_r:", float(escape_r))
    print("white ratio:", float(np.mean(max_mag < float(escape_r))))