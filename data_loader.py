
from __future__ import annotations

import numpy as np
from pathlib import Path
from scipy.io import loadmat


def load_matlab_matrix(path: str, key: str | None = None):
    mat = loadmat(path)
    if key is None:
        key = next(k for k in mat.keys() if not k.startswith("__"))
    return mat[key]

def load_npz_array(path: str | Path, key: str | None = None) -> np.ndarray:
    path = Path(path)
    with np.load(path) as data:
        if key is None:
            key = data.files[0]
        return data[key]

def load_task_npz_pair(
    data_dir: str | Path,
    emotion_file: str = "task-emotion.npz",
    rest_file: str = "task-rest.npz",
    key: str | None = None,
    flatten: bool = True,
):
    data_dir = Path(data_dir)
    X_emotion = load_npz_array(data_dir / emotion_file, key=key)
    X_rest = load_npz_array(data_dir / rest_file, key=key)

    if flatten:
        X_emotion = X_emotion.reshape(X_emotion.shape[0], -1)
        X_rest = X_rest.reshape(X_rest.shape[0], -1)

    return X_emotion, X_rest

def make_dmd_pairs(X: np.ndarray):
    return X[:-1], X[1:]

def generate_state_trajectories(
    A_seq: np.ndarray,
    n_traj: int = 500,
    x0_scale: float = 1.0,
    noise_std: float = 0.0,
    escape_r: float = 2.0,
    c: complex | np.ndarray | None = None,
    c_scale: float = 0.0,
    x0_mode: str = "random",
    use_square: bool = True,
):
    T, n, _ = A_seq.shape
    A_seq = np.asarray(A_seq, dtype=np.float32)

    r = float(escape_r)
    if r <= 0.0:
        r = 2.0

    rng = np.random.default_rng(0)

    if c is None:
        if c_scale > 0.0:
            cr = rng.uniform(-c_scale, c_scale, size=n_traj).astype(np.float32)  # RE
            ci = rng.uniform(-c_scale, c_scale, size=n_traj).astype(np.float32)  # IM
            C_list = (cr + 1j * ci).astype(np.complex64)  # COMPLEX
        else:
            C_list = (np.zeros((n_traj,), dtype=np.complex64))
    elif np.isscalar(c):
        C_list = (np.full((n_traj,), np.complex64(c), dtype=np.complex64))
    else:
        C_list = np.asarray(c, dtype=np.complex64).reshape(-1)
        if C_list.size != n_traj:
            C_list = np.resize(C_list, (n_traj,)).astype(np.complex64)

    X_all, X1_all, X2_all = [], [], []

    for j in range(int(n_traj)):
        if x0_mode == "zeros":
            z = np.zeros((n,), dtype=np.complex64)
        else:
            zr = rng.normal(0.0, 1.0, size=n).astype(np.float32)
            zi = rng.normal(0.0, 1.0, size=n).astype(np.float32)
            z = (x0_scale * (zr + 1j * zi)).astype(np.complex64)

        cj = np.complex64(C_list[j])

        traj = []

        for t in range(int(T)):
            A = A_seq[t]
            Az = (A @ z).astype(np.complex64)

            if use_square:
                z_next = (Az * Az + cj).astype(np.complex64)
            else:
                z_next = (Az + cj).astype(np.complex64)

            # ADD NOISE  # OPTIONAL
            if noise_std > 0.0:  # NOISE
                nr = rng.normal(0.0, noise_std, size=n).astype(np.float32)  # RE
                ni = rng.normal(0.0, noise_std, size=n).astype(np.float32)  # IM
                z_next = (z_next + (nr + 1j * ni).astype(np.complex64)).astype(np.complex64)  # ADD

            mag = np.abs(z_next).astype(np.float32)
            bad = (~np.isfinite(mag)) | (mag > r)
            if np.any(bad):
                mag_safe = np.where((mag > 0.0) & np.isfinite(mag), mag, 1.0).astype(np.float32)
                z_next = np.where(bad, (r / mag_safe) * z_next, z_next).astype(np.complex64)

            xr = z_next.real.astype(np.float32)  # RE
            xi = z_next.imag.astype(np.float32)  # IM
            traj.append(np.concatenate([xr, xi], axis=0).astype(np.float32))

            z = z_next

        X = np.stack(traj, axis=0).astype(np.float32)
        X1, X2 = X[:-1], X[1:]

        X_all.append(X)
        X1_all.append(X1)
        X2_all.append(X2)

    X_all = np.concatenate(X_all, axis=0).astype(np.float32)
    X1_all = np.concatenate(X1_all, axis=0).astype(np.float32)
    X2_all = np.concatenate(X2_all, axis=0).astype(np.float32)

    return X_all, X1_all, X2_all

def load_one_A_matrix(
    data_dir: str | Path,
    source: str = "emotion",
    index: int = 0,
):
    data_dir = Path(data_dir)
    if str(source).lower() == "rest":
        A_all = load_npz_array(data_dir / "task-rest.npz")
    else:
        A_all = load_npz_array(data_dir / "task-emotion.npz")

    A_all = np.asarray(A_all, dtype=np.float32)
    A = A_all[int(index)]
    return A.astype(np.float32)

def load_all_A_matrices(
    data_dir: str | Path,
    source: str = "emotion",
) -> np.ndarray:
    data_dir = Path(data_dir)
    if str(source).lower() == "rest":
        A_all = load_npz_array(data_dir / "task-rest.npz")
    else:
        A_all = load_npz_array(data_dir / "task-emotion.npz")

    A_all = np.asarray(A_all, dtype=np.float32)
    return A_all

def split_explicit_matrix_indices(
    total_count: int,
    train_count: int,
    test_count: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(int(total_count), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(idx)

    train_idx = np.sort(idx[:int(train_count)])
    test_idx = np.sort(idx[int(train_count):int(train_count) + int(test_count)])
    return train_idx, test_idx
