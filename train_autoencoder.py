from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from losses import make_reconstruction_loss, koopman_ae_loss, compute_target_scale

import defines as D

from encoder import Encoder
from decoder import Decoder


def _split_block_train_val(
    X1_block: np.ndarray,
    X2_block: np.ndarray,
    *,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = int(X1_block.shape[0])
    if val_fraction <= 0.0 or n < 2:
        empty = X1_block[:0]
        return X1_block, X2_block, empty, X2_block[:0]

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(round(val_fraction * n)))
    n_val = min(n_val, n - 1)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    return (
        X1_block[train_idx], X2_block[train_idx],
        X1_block[val_idx], X2_block[val_idx],
    )


@torch.no_grad()
def _validation_loss(
    enc, dec, X1_val: np.ndarray, X2_val: np.ndarray, *,
    loss_fn, ridge: float, loss_scale: float, device: torch.device, batch_size: int,
) -> float:
    if X1_val.shape[0] == 0:
        return float("nan")

    X1_cpu = torch.tensor(np.asarray(X1_val, dtype=np.float32))
    X2_cpu = torch.tensor(np.asarray(X2_val, dtype=np.float32))
    loader = DataLoader(TensorDataset(X1_cpu, X2_cpu), batch_size=batch_size, shuffle=False, drop_last=False)

    s = 0.0
    n_batches = 0
    for b1_cpu, b2_cpu in loader:
        b1 = b1_cpu.to(device)
        b2 = b2_cpu.to(device)

        z1 = enc(b1)
        z2 = enc(b2)
        x1_rec = dec(z1)
        x2_rec = dec(z2)

        loss, _ = koopman_ae_loss(
            x1=b1, x2=b2, z1=z1, z2=z2, x1_rec=x1_rec, x2_rec=x2_rec,
            decoder=dec, base_loss=loss_fn,
            alpha_rec=D.AE_REC_WEIGHT, alpha_lin=D.AE_LATENT_WEIGHT, alpha_pred=D.AE_PRED_WEIGHT,
            ridge=ridge, scale=loss_scale,
        )
        s += float(loss.detach().cpu())
        n_batches += 1

    return s / max(1, n_batches)


def _clone_state_dict(model: nn.Module) -> dict:
    return {k: v.clone() if hasattr(v, "clone") else np.array(v, copy=True)
            for k, v in model.state_dict().items()}


def train_autoencoder(
        X1: np.ndarray | list[np.ndarray],
        X2: np.ndarray | list[np.ndarray],
        *,
        latent_dim: int,
        epochs: int,
        batch_size: int,
        lr: float,
        device: torch.device,
) -> tuple[Encoder, Decoder, list[float], dict[str, list[float]], list[float]]:
    if isinstance(X1, np.ndarray):
        X1_list = [X1]
    else:
        X1_list = list(X1)

    if isinstance(X2, np.ndarray):
        X2_list = [X2]
    else:
        X2_list = list(X2)

    if len(X1_list) == 0 or len(X2_list) == 0:
        raise ValueError("X1_list OR X2_list IS EMPTY")
    if len(X1_list) != len(X2_list):
        raise ValueError("X1_list AND X2_list MUST HAVE SAME LENGTH")

    in_dim = int(X1_list[0].shape[1])
    torch.manual_seed(0)
    enc = Encoder(in_dim, latent_dim).to(device)
    dec = Decoder(latent_dim, in_dim).to(device)

    loss_fn = make_reconstruction_loss(0, beta=0.01, w_pow=1.0)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr)


    val_fraction = float(getattr(D, "AE_VAL_FRACTION", 0.0))
    val_seed = int(getattr(D, "AE_VAL_SEED", 0))

    X1_train_list, X2_train_list = [], []
    X1_val_parts, X2_val_parts = [], []

    for bi, (X1_block, X2_block) in enumerate(zip(X1_list, X2_list)):
        x1tr, x2tr, x1va, x2va = _split_block_train_val(
            np.asarray(X1_block, dtype=np.float32), np.asarray(X2_block, dtype=np.float32),
            val_fraction=val_fraction, seed=val_seed + bi,
        )
        X1_train_list.append(x1tr)
        X2_train_list.append(x2tr)
        if x1va.shape[0] > 0:
            X1_val_parts.append(x1va)
            X2_val_parts.append(x2va)

    have_val = len(X1_val_parts) > 0
    if have_val:
        X1_val = np.concatenate(X1_val_parts, axis=0)
        X2_val = np.concatenate(X2_val_parts, axis=0)
    else:
        X1_val = X1_list[0][:0]
        X2_val = X2_list[0][:0]

    n_train_total = int(sum(b.shape[0] for b in X1_train_list))
    n_val_total = int(X1_val.shape[0])
    print(f"[TRAIN AE] train/val split: {n_train_total} train rows, {n_val_total} val rows "
          f"(AE_VAL_FRACTION={val_fraction:g})")

    if bool(getattr(D, "AE_LOSS_AUTO_SCALE", True)):
        loss_scale = compute_target_scale(*X1_train_list, *X2_train_list)
    else:
        loss_scale = 1.0
    print(f"[TRAIN AE] loss_scale={loss_scale:.6e} "
          f"(AE_LOSS_AUTO_SCALE={bool(getattr(D, 'AE_LOSS_AUTO_SCALE', True))})")

    scheduler = None
    if bool(getattr(D, "AE_USE_LR_SCHEDULER", False)):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            patience=int(getattr(D, "AE_LR_PATIENCE", 5)),
            factor=float(getattr(D, "AE_LR_FACTOR", 0.5)),
            min_lr=float(getattr(D, "AE_LR_MIN", 1e-6)),
        )

    early_stop_patience = int(getattr(D, "AE_EARLY_STOP_PATIENCE", 0))
    best_val = float("inf")
    best_state = None
    best_epoch = -1
    bad_epochs = 0

    losses: list[float] = []
    val_losses: list[float] = []
    loss_components: dict[str, list[float]] = {"rec": [], "lin": [], "pred": []}

    for e in range(epochs):
        rng = np.random.default_rng(e)
        order = rng.permutation(len(X1_train_list))

        s = 0.0
        s_rec = 0.0
        s_lin = 0.0
        s_pred = 0.0
        n_batches = 0

        for block_id in order:
            X1_block = np.asarray(X1_train_list[int(block_id)], dtype=np.float32)
            X2_block = np.asarray(X2_train_list[int(block_id)], dtype=np.float32)
            if X1_block.shape[0] == 0:
                continue

            X1_cpu = torch.tensor(X1_block, dtype=torch.float32)
            X2_cpu = torch.tensor(X2_block, dtype=torch.float32)

            loader = DataLoader(
                TensorDataset(X1_cpu, X2_cpu),
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
            )

            for b1_cpu, b2_cpu in loader:
                b1 = b1_cpu.to(device)
                b2 = b2_cpu.to(device)

                z1 = enc(b1)
                z2 = enc(b2)

                x1_rec = dec(z1)
                x2_rec = dec(z2)


                loss, loss_info = koopman_ae_loss(
                    x1=b1,
                    x2=b2,
                    z1=z1,
                    z2=z2,
                    x1_rec=x1_rec,
                    x2_rec=x2_rec,
                    decoder=dec,
                    base_loss=loss_fn,
                    alpha_rec=D.AE_REC_WEIGHT,
                    alpha_lin=D.AE_LATENT_WEIGHT,
                    alpha_pred=D.AE_PRED_WEIGHT,
                    ridge=D.DMD_RIDGE,
                    scale=loss_scale,
                )

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()),
                                               max_norm=1.0)
                opt.step()


                s += float(loss.detach().cpu().item())
                s_rec += float(loss_info["rec"])
                s_lin += float(loss_info["lin"])
                s_pred += float(loss_info["pred"])
                n_batches += 1

        s /= max(1, n_batches)
        losses.append(s)
        loss_components["rec"].append(s_rec / max(1, n_batches))
        loss_components["lin"].append(s_lin / max(1, n_batches))
        loss_components["pred"].append(s_pred / max(1, n_batches))

        enc.eval(); dec.eval()
        v = _validation_loss(
            enc, dec, X1_val, X2_val,
            loss_fn=loss_fn, ridge=D.DMD_RIDGE, loss_scale=loss_scale,
            device=device, batch_size=batch_size,
        )
        enc.train(); dec.train()
        val_losses.append(v)

        monitor = v if have_val else s

        if scheduler is not None:
            lr_before = opt.param_groups[0]["lr"]
            scheduler.step(monitor)
            lr_after = opt.param_groups[0]["lr"]
            if lr_after < lr_before:
                print(f"  [LR SCHEDULER] dropped {lr_before:.3e} -> {lr_after:.3e}")

        if monitor < best_val - 1e-12:
            best_val = monitor
            best_epoch = e + 1
            best_state = (_clone_state_dict(enc), _clone_state_dict(dec))
            bad_epochs = 0
        else:
            bad_epochs += 1

        val_str = f" val={v:.6e}" if have_val else ""
        print(f"EPOCH {e + 1}/{epochs} LOSS={s:.6e}{val_str}  "
              f"[rec={loss_components['rec'][-1]:.3e} lin={loss_components['lin'][-1]:.3e} "
              f"pred={loss_components['pred'][-1]:.3e}]")

        if have_val and early_stop_patience > 0 and bad_epochs >= early_stop_patience:
            print(f"  [EARLY STOP] no val improvement for {bad_epochs} epochs "
                  f"(best {best_val:.6e} @ epoch {best_epoch})")
            break

    if best_state is not None:
        enc.load_state_dict(best_state[0])
        dec.load_state_dict(best_state[1])
        which = "validation" if have_val else "train"
        print(f"[TRAIN AE] restored best checkpoint from epoch {best_epoch} "
              f"({which} loss {best_val:.6e})")

    return enc, dec, losses, loss_components, val_losses