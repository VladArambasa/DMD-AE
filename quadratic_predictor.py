from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

class QuadraticPredictor(nn.Module):
    def __init__(self, state_dim: int, *, rank: int | None = None, bias: bool = False) -> None:
        super().__init__()
        self.state_dim = int(state_dim)

        if rank is None:
            self.A: nn.Module = nn.Linear(self.state_dim, self.state_dim, bias=bias)
        else:
            self.A = nn.Sequential(
                nn.Linear(self.state_dim, int(rank), bias=False),
                nn.Linear(int(rank), self.state_dim, bias=bias),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.state_dim
        zr = x[:, 0:d]
        zi = x[:, d:2 * d]
        cr = x[:, 2 * d:2 * d + 1]
        ci = x[:, 2 * d + 1:2 * d + 2]

        wr = self.A(zr)
        wi = self.A(zi)

        zr_next = wr * wr - wi * wi + cr
        zi_next = 2.0 * wr * wi + ci

        return torch.cat([zr_next, zi_next, cr, ci], dim=1)


def make_quadratic_predictor_pair(
    state_dim: int, *, rank: int | None = None, bias: bool = False,
) -> tuple[nn.Module, nn.Module]:
    return QuadraticPredictor(state_dim, rank=rank, bias=bias), nn.Identity()


def plot_learned_vs_true_matrix_spectrum(
    A_true: np.ndarray,
    A_learned: torch.Tensor,
    out_png,
) -> str:
    import matplotlib.pyplot as plt
    from pathlib import Path

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    A = np.asarray(A_true, dtype=np.float64)
    A_hat = A_learned.detach().cpu().numpy().astype(np.float64)

    A_eigs = np.linalg.eigvals(A)
    A_hat_eigs = np.linalg.eigvals(A_hat)
    A_sigma = np.linalg.svd(A, compute_uv=False)
    A_hat_sigma = np.linalg.svd(A_hat, compute_uv=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(A_eigs.real, A_eigs.imag, "o", label="A (true)")
    ax1.plot(A_hat_eigs.real, A_hat_eigs.imag, "x", label="A (learned)")
    ax1.set_xlabel(r"$\lambda_r$")
    ax1.set_ylabel(r"$\lambda_i$")
    ax1.set_title("Eigenvalues")
    ax1.legend()

    ax2.plot(A_sigma, "o", label=r"$\sigma$(A) (true)")
    ax2.plot(A_hat_sigma, "x", label=r"$\sigma$(A) (learned)")
    ax2.set_title("Singular values")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return str(out_png)