
from __future__ import annotations


from dataclasses import dataclass
from pathlib import Path
import numpy as np

from typing import Optional

from data_loader import (
    load_task_npz_pair,
    generate_state_trajectories,
    load_one_A_matrix,
    load_all_A_matrices,
)


@dataclass
class TrainingData:
    X: np.ndarray
    X1: np.ndarray
    X2: np.ndarray
    meta: dict
    X_grid: Optional[np.ndarray] = None


def _sanitize_finite(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if np.isfinite(x).all():
        return x
    print(f"[WARN] {name} HAD NaN/Inf -> FIXING")
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

def determine_escape_radius(A: np.ndarray) -> float:
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"'A' must be a square matrix: {A.shape}")
    n = int(A.shape[0])
    sigma = np.linalg.svd(A, compute_uv=False)
    sigma_min = float(np.min(sigma))
    if sigma_min <= 0.0:
        return float("inf")
    return float(2.0 * np.sqrt(n) / (sigma_min ** 2))


def _cap_complex_vec(zr: np.ndarray, zi: np.ndarray, r: float) -> tuple[np.ndarray, np.ndarray]:
    r = float(r)
    if r <= 0.0:
        r = 2.0
    r2 = r * r

    mag2 = zr * zr + zi * zi
    over = mag2 > r2
    if np.any(over):
        mag = np.sqrt(np.maximum(mag2, 1e-30)).astype(np.float32, copy=False)
        s = (r / mag).astype(np.float32, copy=False)
        zr = np.where(over, zr * s, zr).astype(np.float32, copy=False)
        zi = np.where(over, zi * s, zi).astype(np.float32, copy=False)
    return zr, zi


def build_from_A_sequences(
    data_dir: str | Path,
    *,
    n_traj: int = 500,
    x0_scale: float = 1.0,
    noise_std: float = 0.0,
    flatten: bool = False,
    escape_r: float = 2.0,
) -> TrainingData:
    data_dir = Path(data_dir)

    A_emotion, A_rest = load_task_npz_pair(data_dir, key=None, flatten=flatten)
    A_seq = np.concatenate([A_emotion, A_rest], axis=0)

    X, X1, X2 = generate_state_trajectories(
        A_seq=A_seq,
        n_traj=n_traj,
        x0_mode="zeros",
        c_scale=0.03,
        use_square=True,
        x0_scale=x0_scale,
        noise_std=noise_std,
        escape_r=escape_r,
    )

    X = _sanitize_finite(X, "X")
    X1 = _sanitize_finite(X1, "X1")
    X2 = _sanitize_finite(X2, "X2")

    return TrainingData(
        X=X.astype(np.float32),
        X1=X1.astype(np.float32),
        X2=X2.astype(np.float32),
        meta={
            "mode": "A_sequences",
            "data_dir": str(data_dir),
            "n_traj": n_traj,
            "x0_scale": x0_scale,
            "noise_std": noise_std,
            "escape_r": float(escape_r),
        },
    )


def build_mandelbrot_training_data(
    *,
    c_re_min: float,
    c_re_max: float,
    c_im_min: float,
    c_im_max: float,
    c_re_n: int = 256,
    c_im_n: int = 256,
    max_iters: int = 40,
    escape_r: float = 2.0,
    seed: int = 0,
) -> TrainingData:

    _ = int(seed)


    T = int(max_iters)
    Nr = int(c_re_n)
    Ni = int(c_im_n)
    r = float(escape_r)
    if r <= 0.0:
        r = 2.0


    cr = np.linspace(float(c_re_min), float(c_re_max), Nr, dtype=np.float32)
    ci = np.linspace(float(c_im_min), float(c_im_max), Ni, dtype=np.float32)
    C_re, C_im = np.meshgrid(cr, ci, indexing="xy")
    P = int(C_re.size)

    cr_flat = C_re.reshape(-1).astype(np.float32, copy=False)
    ci_flat = C_im.reshape(-1).astype(np.float32, copy=False)


    zr = np.zeros((P,), dtype=np.float32)
    zi = np.zeros((P,), dtype=np.float32)

    X_tp4 = np.zeros((T, P, 4), dtype=np.float32)

    for t in range(T):

        zr2 = zr * zr
        zi2 = zi * zi
        zri = 2.0 * zr * zi

        zr = (zr2 - zi2) + cr_flat
        zi = (zri) + ci_flat


        bad = (~np.isfinite(zr)) | (~np.isfinite(zi))
        if np.any(bad):
            zr = np.where(bad, 0.0, zr).astype(np.float32, copy=False)
            zi = np.where(bad, 0.0, zi).astype(np.float32, copy=False)


        zr, zi = _cap_complex_vec(zr, zi, r)


        X_tp4[t, :, 0] = zr
        X_tp4[t, :, 1] = zi
        X_tp4[t, :, 2] = cr_flat
        X_tp4[t, :, 3] = ci_flat



    X_grid = X_tp4.reshape(T, Ni, Nr, 4).astype(np.float32, copy=False)


    X = X_tp4.reshape(T * P, 4).astype(np.float32)

    X1 = X_tp4[:-1].reshape((T - 1) * P, 4).astype(np.float32)
    X2 = X_tp4[1:].reshape((T - 1) * P, 4).astype(np.float32)

    X = _sanitize_finite(X, "X")
    X1 = _sanitize_finite(X1, "X1")
    X2 = _sanitize_finite(X2, "X2")


    print("[TRAINING] X shape     :", X.shape, "dtype:", X.dtype)
    print("[TRAINING] X1 shape    :", X1.shape)
    print("[TRAINING] X2 shape    :", X2.shape)
    print("[TRAINING] X_grid shape:", X_grid.shape, "dtype:", X_grid.dtype)
    print("[TRAINING] zr range:", float(X[:, 0].min()), "to", float(X[:, 0].max()))
    print("[TRAINING] zi range:", float(X[:, 1].min()), "to", float(X[:, 1].max()))
    print("[TRAINING] cr range:", float(X[:, 2].min()), "to", float(X[:, 2].max()))
    print("[TRAINING] ci range:", float(X[:, 3].min()), "to", float(X[:, 3].max()))
    print("[TRAINING] head row 0:", X[0].tolist())

    return TrainingData(
        X=X,
        X1=X1,
        X2=X2,
        meta={
            "mode": "mandelbrot_grid",
            "bbox": (float(c_re_min), float(c_re_max), float(c_im_min), float(c_im_max)),
            "c_re_n": int(Nr),
            "c_im_n": int(Ni),
            "P": int(P),
            "max_iters": int(T),
            "escape_r": float(r),
            "X_size_formula": "max_iters * c_im_n * c_re_n",
        },
        X_grid=X_grid,
    )

def save_training_npz(out_path: str | Path, td: TrainingData) -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if td.X_grid is None:
        np.savez(
            out_path,
            X=td.X,
            X1=td.X1,
            X2=td.X2,
            meta=np.array([td.meta], dtype=object),
        )
    else:
        np.savez(
            out_path,
            X=td.X,
            X1=td.X1,
            X2=td.X2,
            X_grid=td.X_grid,
            meta=np.array([td.meta], dtype=object),
        )
    return str(out_path)


def _build_matrix_c_grid_training_data_from_A(
    A: np.ndarray,
    *,
    c_re_min: float,
    c_re_max: float,
    c_im_min: float,
    c_im_max: float,
    c_re_n: int = 256,
    c_im_n: int = 256,
    max_iters: int = 40,
    escape_r: float = 2.0,
    matrix_index: int | None = None,
    matrix_source: str | None = None,
    filter_escaped: bool = True,
    classify_r: float | None = None,
    keep_escaped_fraction: float = 0.0,
    keep_escaped_seed: int = 0,
) -> TrainingData:
    A = np.asarray(A, dtype=np.float32)

    d = int(A.shape[0])
    T = int(max_iters)
    Nr = int(c_re_n)
    Ni = int(c_im_n)
    r = float(escape_r)
    r2 = r * r

    cr = np.linspace(float(c_re_min), float(c_re_max), Nr, dtype=np.float32)
    ci = np.linspace(float(c_im_min), float(c_im_max), Ni, dtype=np.float32)
    C_re, C_im = np.meshgrid(cr, ci, indexing="xy")
    P = int(C_re.size)

    cr_flat = C_re.reshape(-1).astype(np.float32, copy=False)
    ci_flat = C_im.reshape(-1).astype(np.float32, copy=False)
    c_flat = (cr_flat + 1j * ci_flat).astype(np.complex64)

    z = np.zeros((P, d), dtype=np.complex64)


    feat_dim = 2 * d + 2
    X_tp = np.zeros((T, P, feat_dim), dtype=np.float32)

    for t in range(T):
        Az = (z @ A.T).astype(np.complex64)
        z = (Az * Az).astype(np.complex64)
        z = (z + c_flat[:, None]).astype(np.complex64)

        mag2 = (z.real * z.real + z.imag * z.imag).astype(np.float32)
        bad = (~np.isfinite(mag2)) | (mag2 > r2)
        if np.any(bad):
            mag = np.sqrt(np.maximum(mag2, 1e-30)).astype(np.float32)
            scale = (r / mag).astype(np.float32)
            z_real = np.where(bad, z.real * scale, z.real).astype(np.float32)
            z_imag = np.where(bad, z.imag * scale, z.imag).astype(np.float32)
            z = (z_real + 1j * z_imag).astype(np.complex64)

        X_tp[t, :, 0:d] = z.real.astype(np.float32)
        X_tp[t, :, d:2 * d] = z.imag.astype(np.float32)
        X_tp[t, :, 2 * d] = cr_flat
        X_tp[t, :, 2 * d + 1] = ci_flat

    X_grid = X_tp.reshape(T, Ni, Nr, feat_dim).astype(np.float32, copy=False)
    X = X_tp.reshape(T * P, feat_dim).astype(np.float32)


    c_r = float(escape_r if classify_r is None else classify_r)
    final_zr = X_tp[-1, :, 0:d]
    final_zi = X_tp[-1, :, d:2 * d]
    final_mag2 = np.max(final_zr * final_zr + final_zi * final_zi, axis=1)
    alive = np.isfinite(final_mag2) & (final_mag2 < c_r * c_r)

    if bool(filter_escaped) and int(np.count_nonzero(alive)) > 0:
        keep = alive.copy()

        frac = float(keep_escaped_fraction)
        if frac > 0.0:
            escaped_idx = np.flatnonzero(~alive)
            n_keep = int(round(frac * escaped_idx.size))
            if n_keep > 0:
                rng = np.random.default_rng(int(keep_escaped_seed))
                chosen = rng.choice(escaped_idx, size=min(n_keep, escaped_idx.size), replace=False)
                keep[chosen] = True

        X1 = X_tp[:-1][:, keep, :].reshape(-1, feat_dim).astype(np.float32)
        X2 = X_tp[1:][:, keep, :].reshape(-1, feat_dim).astype(np.float32)
    else:
        if bool(filter_escaped):
            print(f"[WARN] filter_escaped=True BUT 0/{P} POINTS SURVIVED AT classify_r={c_r} "
                  f"-- FALLING BACK TO THE UNFILTERED GRID FOR THIS MATRIX")
        X1 = X_tp[:-1].reshape((T - 1) * P, feat_dim).astype(np.float32)
        X2 = X_tp[1:].reshape((T - 1) * P, feat_dim).astype(np.float32)

    X = _sanitize_finite(X, "X")
    X1 = _sanitize_finite(X1, "X1")
    X2 = _sanitize_finite(X2, "X2")

    n_alive = int(np.count_nonzero(alive))
    n_kept = int(X1.shape[0] // max(T - 1, 1))
    extra = f", +{n_kept - n_alive} ESCAPED (keep_escaped_fraction={keep_escaped_fraction:g})" if n_kept > n_alive else ""
    print(f"[TRAINING] matrix_c_grid: {n_alive}/{P} points bounded at classify_r={c_r:g} "
          f"(clamp_r={r:g}) -> X1/X2 rows={X1.shape[0]}{extra}"
          + (" (UNFILTERED)" if not filter_escaped else ""))

    return TrainingData(
        X=X,
        X1=X1,
        X2=X2,
        meta={
            "mode": "matrix_c_grid",
            "matrix_index": None if matrix_index is None else int(matrix_index),
            "matrix_source": matrix_source,
            "state_dim": int(d),
            "max_iters": int(T),
            "c_re_n": int(Nr),
            "c_im_n": int(Ni),
            "escape_r": float(r),
            "classify_r": float(c_r),
            "filter_escaped": bool(filter_escaped),
            "keep_escaped_fraction": float(keep_escaped_fraction),
            "n_alive": n_alive,
            "n_kept": n_kept,
            "n_total": int(P),
            "alive_mask_grid": alive.reshape(Ni, Nr),
        },
        X_grid=X_grid,
    )

def build_matrix_c_grid_training_data(
    data_dir: str | Path,
    *,
    source: str = "emotion",
    index: int = 0,
    c_re_min: float,
    c_re_max: float,
    c_im_min: float,
    c_im_max: float,
    c_re_n: int = 256,
    c_im_n: int = 256,
    max_iters: int = 40,
    escape_r: float = 2.0,
    filter_escaped: bool = True,
    classify_r: float | None = None,
    keep_escaped_fraction: float = 0.0,
) -> TrainingData:
    data_dir = Path(data_dir)
    A = load_one_A_matrix(data_dir, source=source, index=index)
    return _build_matrix_c_grid_training_data_from_A(
        A,
        c_re_min=c_re_min,
        c_re_max=c_re_max,
        c_im_min=c_im_min,
        c_im_max=c_im_max,
        c_re_n=c_re_n,
        c_im_n=c_im_n,
        max_iters=max_iters,
        escape_r=escape_r,
        matrix_index=index,
        matrix_source=source,
        filter_escaped=filter_escaped,
        classify_r=classify_r,
        keep_escaped_fraction=keep_escaped_fraction,
    )

def build_matrix_c_grid_training_data_many_matrices(
    data_dir: str | Path,
    *,
    source: str = "emotion",
    indices: list[int] | np.ndarray,
    c_re_min: float,
    c_re_max: float,
    c_im_min: float,
    c_im_max: float,
    c_re_n: int = 8,
    c_im_n: int = 8,
    max_iters: int = 40,
    escape_r: float = 2.0,
    filter_escaped: bool = True,
    classify_r: float | None = None,
) -> list[TrainingData]:
    data_dir = Path(data_dir)
    A_all = load_all_A_matrices(data_dir, source=source)
    out: list[TrainingData] = []

    for idx in np.asarray(indices, dtype=np.int64):
        td = _build_matrix_c_grid_training_data_from_A(
            A_all[int(idx)],
            c_re_min=c_re_min,
            c_re_max=c_re_max,
            c_im_min=c_im_min,
            c_im_max=c_im_max,
            c_re_n=c_re_n,
            c_im_n=c_im_n,
            max_iters=max_iters,
            escape_r=escape_r,
            matrix_index=int(idx),
            matrix_source=source,
            filter_escaped=filter_escaped,
            classify_r=classify_r,
        )
        out.append(td)

    return out
