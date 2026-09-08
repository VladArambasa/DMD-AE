from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import defines as D



# SHARED PLUMBING

class StepwiseLatentOperator:

    def __init__(self, one_step_fn: Callable[[torch.Tensor], torch.Tensor], device="cpu"):
        self.one_step_fn = one_step_fn
        self.device = device
        self.A: Optional[torch.Tensor] = None

    @torch.no_grad()
    def predict(self, z0: torch.Tensor, steps: int = 20) -> torch.Tensor:
        preds = []
        z = z0.clone().to(self.device)
        for _ in range(steps):
            z = self.one_step_fn(z)
            preds.append(z.clone())
        return torch.stack(preds)


def anae(ref: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
    ref_flat = ref.reshape(-1)
    new_flat = new.reshape(-1)
    ratio = torch.abs(ref_flat - new_flat) / torch.abs(ref_flat)
    ratio = ratio[torch.isfinite(ratio)]
    if ratio.numel() == 0:
        return torch.tensor(float("nan"))
    return 100.0 * ratio.mean()


class _ForwardView(nn.Module):

    def __init__(self, module: nn.Module, method_name: str):
        super().__init__()
        self.wrapped = module
        self._method_name = method_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return getattr(self.wrapped, self._method_name)(x)


def _split_rows_train_val(
    X1: np.ndarray, X2: np.ndarray, *, val_fraction: float, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    n = int(X1.shape[0])
    if val_fraction <= 0.0 or n < 2:
        empty = X1[:0]
        return X1, X2, empty, X2[:0]

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(round(val_fraction * n)))
    n_val = min(n_val, n - 1)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return X1[train_idx], X2[train_idx], X1[val_idx], X2[val_idx]


def _clone_module_state(module: nn.Module) -> dict:
    return {k: v.clone() for k, v in module.state_dict().items()}


def run_koopman_training_loop(
    X1: np.ndarray,
    X2: np.ndarray,
    *,
    bundle: nn.Module,
    step_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor], tuple[torch.Tensor, dict]],
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    label: str,
) -> tuple[list[float], dict[str, list[float]], list[float]]:

    bundle = bundle.to(device)
    opt = torch.optim.Adam(bundle.parameters(), lr=lr)

    val_fraction = float(getattr(D, "AE_VAL_FRACTION", 0.0))
    val_seed = int(getattr(D, "AE_VAL_SEED", 0))
    X1_tr, X2_tr, X1_va, X2_va = _split_rows_train_val(
        np.asarray(X1, dtype=np.float32), np.asarray(X2, dtype=np.float32),
        val_fraction=val_fraction, seed=val_seed,
    )
    have_val = X1_va.shape[0] > 0
    print(f"[{label}] train/val split: {X1_tr.shape[0]} train rows, {X1_va.shape[0]} val rows "
          f"(AE_VAL_FRACTION={val_fraction:g})")

    scheduler = None
    if bool(getattr(D, "AE_USE_LR_SCHEDULER", False)):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min",
            patience=int(getattr(D, "AE_LR_PATIENCE", 5)),
            factor=float(getattr(D, "AE_LR_FACTOR", 0.5)),
            min_lr=float(getattr(D, "AE_LR_MIN", 1e-6)),
        )
    early_stop_patience = int(getattr(D, "AE_EARLY_STOP_PATIENCE", 0))

    best_val = float("inf")
    best_state: Optional[dict] = None
    best_epoch = -1
    bad_epochs = 0

    losses: list[float] = []
    val_losses: list[float] = []
    loss_components: dict[str, list[float]] = {}

    X1_tr_cpu = torch.tensor(X1_tr, dtype=torch.float32)
    X2_tr_cpu = torch.tensor(X2_tr, dtype=torch.float32)

    for e in range(epochs):
        loader = DataLoader(
            TensorDataset(X1_tr_cpu, X2_tr_cpu), batch_size=batch_size, shuffle=True, drop_last=False,
        )
        sums: dict[str, float] = {}
        n_batches = 0

        for b1_cpu, b2_cpu in loader:
            b1 = b1_cpu.to(device)
            b2 = b2_cpu.to(device)

            loss, info = step_fn(bundle, b1, b2)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bundle.parameters(), max_norm=1.0)
            opt.step()

            for k, v in info.items():
                sums[k] = sums.get(k, 0.0) + float(v)
            sums["total"] = sums.get("total", 0.0) + float(loss.detach().cpu())
            n_batches += 1

        n_batches = max(1, n_batches)
        epoch_mean = {k: v / n_batches for k, v in sums.items()}
        losses.append(epoch_mean.get("total", float("nan")))
        for k, v in epoch_mean.items():
            if k == "total":
                continue
            loss_components.setdefault(k, []).append(v)


        bundle.eval()
        v = float("nan")
        if have_val:
            with torch.no_grad():
                X1v = torch.tensor(X1_va, dtype=torch.float32, device=device)
                X2v = torch.tensor(X2_va, dtype=torch.float32, device=device)
                vloss, _ = step_fn(bundle, X1v, X2v)
                v = float(vloss.detach().cpu())
        bundle.train()
        val_losses.append(v)

        monitor = v if have_val else epoch_mean.get("total", float("inf"))

        if scheduler is not None:
            lr_before = opt.param_groups[0]["lr"]
            scheduler.step(monitor)
            lr_after = opt.param_groups[0]["lr"]
            if lr_after < lr_before:
                print(f"  [{label}][LR SCHEDULER] dropped {lr_before:.3e} -> {lr_after:.3e}")

        if monitor < best_val - 1e-12:
            best_val = monitor
            best_epoch = e + 1
            best_state = _clone_module_state(bundle)
            bad_epochs = 0
        else:
            bad_epochs += 1

        val_str = f" val={v:.6e}" if have_val else ""
        print(f"[{label}] EPOCH {e + 1}/{epochs} LOSS={epoch_mean.get('total', float('nan')):.6e}{val_str}")

        if have_val and early_stop_patience > 0 and bad_epochs >= early_stop_patience:
            print(f"  [{label}][EARLY STOP] no val improvement for {bad_epochs} epochs "
                  f"(best {best_val:.6e} @ epoch {best_epoch})")
            break

    if best_state is not None:
        bundle.load_state_dict(best_state)
        which = "validation" if have_val else "train"
        print(f"[{label}] restored best checkpoint from epoch {best_epoch} ({which} loss {best_val:.6e})")

    return losses, loss_components, val_losses

class LuschKoopmanOperator(nn.Module):

    def __init__(self, latent_dim: int, delta_t: float = 0.01):
        super().__init__()
        if latent_dim % 2 != 0:
            raise ValueError(
                f"LuschKoopmanOperator needs an EVEN latent_dim (got {latent_dim}): "
                "the reference architecture pairs up latent coordinates into "
                "complex-conjugate 2-D rotation-scaling blocks."
            )
        self.latent_dim = int(latent_dim)
        self.num_pairs = self.latent_dim // 2
        self.delta_t = float(delta_t)

        self.parameterization = nn.Sequential(
            nn.Linear(self.latent_dim, self.num_pairs * 2),
            nn.Tanh(),
            nn.Linear(self.num_pairs * 2, self.num_pairs * 2),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        unbatched = y.dim() == 1
        if unbatched:
            y = y.unsqueeze(0)

        mu, omega = torch.unbind(self.parameterization(y).reshape(-1, self.num_pairs, 2), -1)
        exp_ = torch.exp(self.delta_t * mu)
        cos_ = torch.cos(self.delta_t * omega)
        sin_ = torch.sin(self.delta_t * omega)

        y_pairs = y.reshape(-1, self.num_pairs, 2)
        yr, yi = y_pairs[..., 0], y_pairs[..., 1]

        yr_next = (cos_ * yr - sin_ * yi) * exp_
        yi_next = (sin_ * yr + cos_ * yi) * exp_
        y_next = torch.stack([yr_next, yi_next], dim=-1).reshape(-1, self.latent_dim)

        return y_next.squeeze(0) if unbatched else y_next


class LuschKoopmanAE(nn.Module):

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int, delta_t: float = 0.01):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.koopman = LuschKoopmanOperator(latent_dim, delta_t=delta_t)

        self.register_buffer("mu", torch.zeros(input_dim))
        self.register_buffer("std", torch.ones(input_dim))

    def set_normalization(self, X: torch.Tensor) -> None:
        with torch.no_grad():
            self.mu.copy_(X.mean(dim=0))
            self.std.copy_(torch.clamp(X.std(dim=0), min=1e-6))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder((x - self.mu) / self.std)

    def recover(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z) * self.std + self.mu

    def koopman_step(self, z: torch.Tensor) -> torch.Tensor:
        return self.koopman(z)


def lusch_koopman_ae_loss(
    ae: LuschKoopmanAE, x1: torch.Tensor, x2: torch.Tensor, *, alpha1: float = 2.0, alpha2: float = 1e-8,
) -> tuple[torch.Tensor, dict]:

    z1 = ae.embed(x1)
    z2 = ae.embed(x2)
    x1_rec = ae.recover(z1)
    x2_rec = ae.recover(z2)
    z2_pred = ae.koopman_step(z1)
    x2_pred = ae.recover(z2_pred)

    rec = F.mse_loss(x1_rec, x1) + F.mse_loss(x2_rec, x2)
    lin = F.mse_loss(z2_pred, z2)
    pred = F.mse_loss(x2_pred, x2)

    inf_rec = (torch.norm(x1 - x1_rec, p=float("inf"), dim=-1).mean()
               + torch.norm(x2 - x2_rec, p=float("inf"), dim=-1).mean())
    inf_pred = torch.norm(x2 - x2_pred, p=float("inf"), dim=-1).mean()
    inf_term = inf_rec + inf_pred

    total = alpha1 * (pred + rec) + lin + alpha2 * inf_term
    info = {
        "rec": float(rec.detach().cpu()), "lin": float(lin.detach().cpu()),
        "pred": float(pred.detach().cpu()), "inf": float(inf_term.detach().cpu()),
    }
    return total, info


def train_lusch(
    X1: np.ndarray, X2: np.ndarray, *,
    latent_dim: int, epochs: int, batch_size: int, lr: float, device: torch.device,
    hidden_dim: Optional[int] = None, delta_t: float = 0.01,
    alpha1: float = 2.0, alpha2: float = 1e-8,
) -> tuple[nn.Module, nn.Module, StepwiseLatentOperator, list[float], dict[str, list[float]], list[float]]:
    if latent_dim % 2 != 0:
        latent_dim += 1
        print(f"[LUSCH] latent_dim bumped to {latent_dim} (must be even -- see LuschKoopmanOperator)")

    in_dim = int(X1.shape[1])
    if hidden_dim is None:
        hidden_dim = 512

    torch.manual_seed(0)
    ae = LuschKoopmanAE(in_dim, latent_dim, hidden_dim, delta_t=delta_t)
    ae.set_normalization(torch.tensor(np.asarray(X1, dtype=np.float32)))

    def step_fn(bundle: LuschKoopmanAE, b1, b2):
        return lusch_koopman_ae_loss(bundle, b1, b2, alpha1=alpha1, alpha2=alpha2)

    losses, loss_components, val_losses = run_koopman_training_loop(
        X1, X2, bundle=ae, step_fn=step_fn, epochs=epochs, batch_size=batch_size,
        lr=lr, device=device, label="LUSCH",
    )

    encoder = _ForwardView(ae, "embed")
    decoder = _ForwardView(ae, "recover")
    dmd = StepwiseLatentOperator(lambda z: ae.koopman_step(z), device=device)

    return encoder, decoder, dmd, losses, loss_components, val_losses

class MLPStack(nn.Module):

    def __init__(self, input_size: int, output_size: int, hidden_sizes: list[int], batch_norm: bool = False):
        super().__init__()
        sizes = [input_size, *hidden_sizes, output_size]
        layers: list[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i != len(sizes) - 2:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(sizes[i + 1]))
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DLKoopmanAutoEncoder(nn.Module):

    def __init__(
        self, input_size: int, encoded_size: int,
        encoder_hidden_layers: list[int], decoder_hidden_layers: Optional[list[int]] = None,
        batch_norm: bool = False,
    ):
        super().__init__()
        if not decoder_hidden_layers:
            decoder_hidden_layers = list(reversed(encoder_hidden_layers))
        self.encoder = MLPStack(input_size, encoded_size, list(encoder_hidden_layers), batch_norm)
        self.decoder = MLPStack(encoded_size, input_size, list(decoder_hidden_layers), batch_norm)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return z, self.decoder(z)


class KoopmanLinearOperator(nn.Module):

    def __init__(self, size: int):
        super().__init__()
        self.net = nn.Linear(size, size, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class _DLKoopmanBundle(nn.Module):

    def __init__(self, ae: DLKoopmanAutoEncoder, knet: KoopmanLinearOperator):
        super().__init__()
        self.ae = ae
        self.knet = knet


def dlkoopman_loss(
    ae: DLKoopmanAutoEncoder, knet: KoopmanLinearOperator, x1: torch.Tensor, x2: torch.Tensor,

        *, decoder_loss_weight: float = 1.0
) -> tuple[torch.Tensor, dict]:

    z1, x1_rec = ae(x1)
    z2, x2_rec = ae(x2)
    z2_pred = knet(z1)
    x2_pred = ae.decoder(z2_pred)

    recon = F.mse_loss(x1_rec, x1) + F.mse_loss(x2_rec, x2)
    lin = F.mse_loss(z2_pred, z2)
    pred = F.mse_loss(x2_pred, x2)
    total = lin + decoder_loss_weight * (recon + pred)

    info = {"rec": float(recon.detach().cpu()), "lin": float(lin.detach().cpu()), "pred": float(pred.detach().cpu())}
    return total, info


def train_dlkoopman(
    X1: np.ndarray, X2: np.ndarray, *,
    latent_dim: int, epochs: int, batch_size: int, lr: float, device: torch.device,
    encoder_hidden_layers: Optional[list[int]] = None, decoder_hidden_layers: Optional[list[int]] = None,
    batch_norm: bool = False,

    decoder_loss_weight: float = 1.0
) -> tuple[nn.Module, nn.Module, StepwiseLatentOperator, list[float], dict[str, list[float]], list[float]]:

    if encoder_hidden_layers is None:
        encoder_hidden_layers = [100]

    in_dim = int(X1.shape[1])
    torch.manual_seed(0)
    ae = DLKoopmanAutoEncoder(in_dim, latent_dim, encoder_hidden_layers, decoder_hidden_layers, batch_norm)
    knet = KoopmanLinearOperator(latent_dim)
    bundle = _DLKoopmanBundle(ae, knet)

    def step_fn(b: _DLKoopmanBundle, b1, b2):
        return dlkoopman_loss(b.ae, b.knet, b1, b2, decoder_loss_weight=decoder_loss_weight)

    losses, loss_components, val_losses = run_koopman_training_loop(
        X1, X2, bundle=bundle, step_fn=step_fn, epochs=epochs, batch_size=batch_size,
        lr=lr, device=device, label="DLKOOPMAN",
    )

    encoder = bundle.ae.encoder
    decoder = bundle.ae.decoder
    dmd = StepwiseLatentOperator(lambda z: bundle.knet(z), device=device)
    dmd.A = bundle.knet.net.weight.detach()

    return encoder, decoder, dmd, losses, loss_components, val_losses

class DLDMDAutoEncoder(nn.Module):

    def __init__(self, input_dim: int, latent_dim: int, hidden_sizes: tuple[int, ...] = (128, 128),
                 l1_coefficient: float = 0.01):
        super().__init__()
        self.l1_coefficient = float(l1_coefficient)
        self.encoder = nn.Sequential(*self._make_layers([input_dim, *hidden_sizes, latent_dim]))
        self.decoder = nn.Sequential(*self._make_layers([latent_dim, *hidden_sizes, input_dim]))

    @staticmethod
    def _make_layers(sizes: list[int]) -> list[nn.Module]:
        layers: list[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i != len(sizes) - 2:
                layers.append(nn.Tanh())
        return layers

    def l1_penalty(self) -> torch.Tensor:
        total = torch.zeros((), device=next(self.parameters()).device)
        for block in (self.encoder, self.decoder):
            for layer in block:
                if isinstance(layer, nn.Linear):
                    total = total + layer.weight.abs().sum()
        return self.l1_coefficient * total


class ExactDMDDynamics:

    def __init__(self, device="cpu", rank: Optional[int] = None):
        self.A: Optional[torch.Tensor] = None
        self.device = device
        self.rank = rank
        self.eigenvalues: Optional[torch.Tensor] = None
        self.svd_U: Optional[torch.Tensor] = None

    def fit(self, Z1: torch.Tensor, Z2: torch.Tensor) -> torch.Tensor:
        Z1_t = Z1 if torch.is_tensor(Z1) else torch.as_tensor(Z1, dtype=torch.float32, device=self.device)
        Z2_t = Z2 if torch.is_tensor(Z2) else torch.as_tensor(Z2, dtype=torch.float32, device=self.device)

        X_minus = Z1_t.T.double()
        X_plus = Z2_t.T.double()

        U, S, Vh = torch.linalg.svd(X_minus, full_matrices=False)
        k = int(S.shape[0])
        r = max(1, min(int(self.rank), k)) if self.rank else k
        U_r, S_r, Vh_r = U[:, :r], S[:r], Vh[:r, :]

        tol = S_r[0] * 1e-6 if S_r.numel() > 0 else 0.0
        S_inv = torch.where(S_r > tol, 1.0 / S_r, torch.zeros_like(S_r))

        pinv_X_minus = (Vh_r.T * S_inv) @ U_r.T
        Atilde = X_plus @ pinv_X_minus

        self.A = Atilde.T.to(Z1_t.dtype)
        self.svd_U = U_r.to(Z1_t.dtype)
        try:
            self.eigenvalues = torch.linalg.eigvals(Atilde)
        except Exception:
            self.eigenvalues = None
        return self.A

    def predict(self, z0: torch.Tensor, steps: int = 20) -> torch.Tensor:
        preds = []
        z = z0.clone().to(self.device)
        for _ in range(steps):
            z = (self.A @ z) if z.ndim == 1 else (z @ self.A.T)
            preds.append(z.clone())
        return torch.stack(preds)

    def subspace_residual(self, z: torch.Tensor) -> torch.Tensor:

        if self.svd_U is None:
            return torch.zeros((), device=z.device)
        U = self.svd_U.to(z.device)
        residual = z - (z @ U) @ U.T
        return residual.pow(2).sum(dim=-1).mean()


def dldmd_loss(
    ae: DLDMDAutoEncoder, dmd: ExactDMDDynamics, x1: torch.Tensor, x2: torch.Tensor,
    z1: torch.Tensor, z2: torch.Tensor, *, c1: float = 1.0, c2: float = 1.0, c3: float = 1.0,
) -> tuple[torch.Tensor, dict]:

    x1_rec = ae.decoder(z1)
    x2_rec = ae.decoder(z2)
    z2_pred = z1 @ dmd.A.T
    x2_pred = ae.decoder(z2_pred)

    ae_loss = F.mse_loss(x1_rec, x1) + F.mse_loss(x2_rec, x2)
    predict_loss = F.mse_loss(z2_pred, z2)
    linearity_loss = F.mse_loss(x2_pred, x2)
    subspace = dmd.subspace_residual(z2)
    l1 = ae.l1_penalty()

    total = c1 * ae_loss + c2 * predict_loss + c3 * linearity_loss + subspace + l1
    info = {
        "rec": float(ae_loss.detach().cpu()), "lin": float(predict_loss.detach().cpu()),
        "pred": float(linearity_loss.detach().cpu()),
        "subspace": float(subspace.detach().cpu()), "l1": float(l1.detach().cpu()),
    }
    return total, info


def train_dldmd(
    X1: np.ndarray, X2: np.ndarray, *,
    latent_dim: int, epochs: int, batch_size: int, lr: float, device: torch.device,
    hidden_sizes: tuple[int, ...] = (128, 128), l1_coefficient: float = 1e-6,
    c1: float = 1.0, c2: float = 1.0, c3: float = 1.0, dmd_rank: Optional[int] = None,
) -> tuple[nn.Module, nn.Module, ExactDMDDynamics, list[float], dict[str, list[float]], list[float]]:

    in_dim = int(X1.shape[1])
    torch.manual_seed(0)
    ae = DLDMDAutoEncoder(in_dim, latent_dim, hidden_sizes, l1_coefficient).to(device)
    dmd = ExactDMDDynamics(device=device, rank=dmd_rank)

    def step_fn(bundle: DLDMDAutoEncoder, b1, b2):
        z1 = bundle.encoder(b1)
        z2 = bundle.encoder(b2)
        dmd.fit(z1, z2)
        return dldmd_loss(bundle, dmd, b1, b2, z1, z2, c1=c1, c2=c2, c3=c3)

    losses, loss_components, val_losses = run_koopman_training_loop(
        X1, X2, bundle=ae, step_fn=step_fn, epochs=epochs, batch_size=batch_size,
        lr=lr, device=device, label="DLDMD",
    )

    with torch.no_grad():
        Z1 = ae.encoder(torch.tensor(np.asarray(X1, dtype=np.float32), device=device))
        Z2 = ae.encoder(torch.tensor(np.asarray(X2, dtype=np.float32), device=device))
        dmd.fit(Z1, Z2)

    return ae.encoder, ae.decoder, dmd, losses, loss_components, val_losses

ARCHITECTURES: dict[str, Callable] = {
    "lusch": train_lusch,
    "dlkoopman": train_dlkoopman,
    "dldmd": train_dldmd,
}

ARCHITECTURE_CITATIONS: dict[str, str] = {
    "lusch": (
        "B. Lusch, J.N. Kutz, S.L. Brunton, \"Deep learning for universal linear "
        "embeddings of nonlinear dynamics\", Nat. Commun. 9, 4950 (2018). Auxiliary "
        "network parameterises a continuous complex-eigenvalue Koopman operator "
        "(closed-form per-step rotation+scaling, never fit by regression, never a "
        "free learned matrix). Reference: Python_Code_notMine_DeepKoopmanLusch_master.txt."
    ),
    "dlkoopman": (
        "S. Dey et al., the `dlkoopman` package (TrajPred model). Koopman matrix is "
        "a single nn.Linear layer trained jointly with the autoencoder via backprop. "
        "Reference: Python_Code_notMine_dlkoopman_main.txt."
    ),
    "dldmd": (
        "O. Issan et al., \"DLDMD\". Exact SVD-pseudoinverse DMD (no ridge) refit on "
        "encoded snapshot pairs every batch, plus L1-regularised encoder/decoder "
        "weights and a DMD-subspace residual loss term. Reference: "
        "Python_Code_notMine_opaliss.txt / Python_Code_notMine_dmd_autoencoder_main.txt "
        "(identical files)."
    ),
}