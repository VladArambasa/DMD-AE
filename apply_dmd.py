from __future__ import annotations

import numpy as np
import torch

from latent_dynamics import DMDDynamics

def fit_dmd_on_arrays(X1: np.ndarray, X2: np.ndarray, *, device: torch.device, ridge: float = 1e-2) -> DMDDynamics:

    dmd = DMDDynamics(device=device)
    dmd.fit(X1, X2, ridge=ridge)
    assert dmd.A is not None, "DMDDynamics.fit did not set self.A"
    return dmd

@torch.no_grad()
def predict_orbit(dmd: DMDDynamics, z0: torch.Tensor, *, steps: int) -> torch.Tensor:
    return dmd.predict(z0, steps=steps)

def export_dmd_matrix(dmd: DMDDynamics, out_npy: str, out_csv: str | None = None) -> None:
    A = getattr(dmd, "A", None)
    if A is None:
        raise RuntimeError("DMD is not fitted (A is None).")

    A_cpu = A.detach().cpu().numpy()
    np.save(out_npy, A_cpu)
    if out_csv is not None:
        np.savetxt(out_csv, A_cpu, delimiter=",")

def fit_dmd_from_latent_covariances(
    G: np.ndarray,
    H: np.ndarray,
    *,
    device: torch.device,
    ridge: float = 1e-2,
) -> DMDDynamics:

    A_t = np.linalg.solve(G + ridge * np.eye(G.shape[0], dtype=G.dtype), H)
    dmd = DMDDynamics(device=device)
    dmd.A = torch.tensor(A_t.T, dtype=torch.float32, device=device)
    return dmd