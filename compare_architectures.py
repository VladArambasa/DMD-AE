from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import defines as D
from run_single_matrix import run_single_matrix
from data_loader import load_one_A_matrix
from prepare_training_data import build_matrix_c_grid_training_data, save_training_npz, determine_escape_radius
from encoder import Encoder
from decoder import Decoder
from apply_dmd import fit_dmd_on_arrays
from eval_matrix_dmd_ae import (
    save_loss_curve,
    autoencoder_reconstruction_metrics,
    autoencoder_reconstruction_metrics_alive,
    dmd_one_step_metrics,
    save_ground_truth_final_mask,
    save_ground_truth_escape_iters,
    reconstruct_true_final_snapshot,
    iterate_true_next_snapshot,
    predict_next_snapshot,
    next_step_prediction_metrics,
    predict_rollout_from_start_ae_dmd,
    teacher_forced_escape_iters,
)
from mandelbrot_reconstruct import save_final_snapshot_image, save_escape_image
from experiment_common import pick_device, print_metric_block, write_metrics_txt, debug_final_state_stats
from utils import save_model, load_model, to_tensor, set_recommended_matplotlib

import other_architectures as OA


COMPARISON_DIRNAME = "aa__comparison_output__aa"

def _arch_dirs(name: str) -> dict:
    root = Path("out") / "single-matrix" if name == "mine" else Path("out") / name / "single-matrix"
    return {"root": root, "td": root / "training-data", "res": root / "results"}


def _make_arch_out_dirs(name: str) -> dict:
    dirs = _arch_dirs(name)
    dirs["td"].mkdir(parents=True, exist_ok=True)
    dirs["res"].mkdir(parents=True, exist_ok=True)
    return dirs


def _read_metrics_txt(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            try:
                metrics[k.strip()] = float(v.strip())
            except ValueError:
                continue
    return metrics


def _extract_rollout_curve(metrics: dict, *, alive: bool = False) -> list[float]:
    prefix = "rollout_rel_l2_alive_step_" if alive else "rollout_rel_l2_step_"
    pairs = sorted(
        (int(k[len(prefix):]), v) for k, v in metrics.items()
        if k.startswith(prefix) and k[len(prefix):].isdigit()
    )
    return [v for _, v in pairs]

def _save_fullgrid_comparison_images(name: str, td, enc, dec, dmd, device: torch.device, res_dir: Path) -> None:
    Z_recon = reconstruct_true_final_snapshot(td, enc, dec, device)
    save_final_snapshot_image(Z_recon, escape_r=D.ESCAPE_R, out_png=res_dir / "recon_final_mask_fullgrid.png",
                               mode="mask")

    maxit = int(D.TRAIN_MAX_ITERS)
    rollout = predict_rollout_from_start_ae_dmd(td, enc, dec, dmd, device, steps=maxit, escape_r=D.DYNAMICS_CLAMP_R)
    Z_pred_final = rollout[-1]
    save_final_snapshot_image(Z_pred_final, escape_r=D.ESCAPE_R,
                               out_png=res_dir / "rollout_from_start_final_mask_fullgrid.png", mode="mask")

    iters_pred = teacher_forced_escape_iters(td, enc, dec, dmd, device, escape_r=D.ESCAPE_R)
    save_escape_image(iters_pred, max_iters=maxit, out_png=res_dir / "pred_escape_iters_fullgrid.png")
    print(f"  [{name}] full-grid (untinted) comparison images saved to {res_dir}")


def reload_mine_model(td, device: torch.device) -> tuple:
    in_dim = int(td.X1.shape[1])
    enc = Encoder(in_dim, D.LATENT_DIM).to(device)
    dec = Decoder(D.LATENT_DIM, in_dim).to(device)
    load_model(enc, Path(D.CHECKPOINT_DIR) / "encoder_single_matrix.pth", device)
    load_model(dec, Path(D.CHECKPOINT_DIR) / "decoder_single_matrix.pth", device)
    enc.eval()
    dec.eval()

    with torch.no_grad():
        Z1 = enc(to_tensor(td.X1, device)).detach().cpu().numpy()
        Z2 = enc(to_tensor(td.X2, device)).detach().cpu().numpy()
    dmd = fit_dmd_on_arrays(Z1, Z2, device=device, ridge=D.DMD_RIDGE)

    return enc, dec, dmd


def run_mine_experiment(device: torch.device) -> dict:
    print("\n---------------- TRAINING: mine (via run_single_matrix.py, unmodified) ----------------")
    run_single_matrix(device)
    dirs = _arch_dirs("mine")
    metrics_path = dirs["res"] / "metrics.txt"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"run_single_matrix() finished but {metrics_path} wasn't created -- "
            "can't read back 'mine's metrics for the comparison."
        )
    return _read_metrics_txt(metrics_path)

def run_full_experiment_for_architecture(
    name: str, trainer, td, A: np.ndarray, device: torch.device, *,
    latent_dim: int, epochs: int, batch_size: int, lr: float,
) -> tuple[dict, list[float]]:
    dirs = _make_arch_out_dirs(name)
    print(f"\n---------------- TRAINING: {name} ----------------")
    if name in OA.ARCHITECTURE_CITATIONS:
        print(f"  ({OA.ARCHITECTURE_CITATIONS[name]})")

    save_training_npz(dirs["td"] / "training_single_matrix.npz", td)
    save_ground_truth_escape_iters(td, D.ESCAPE_R, dirs["td"] / "gt_escape_iters.png")
    save_ground_truth_final_mask(td, D.ESCAPE_R, dirs["td"] / "gt_final_mask.png", scale=D.IMAGE_SCALE)

    try:
        principled_r = determine_escape_radius(A)
        print(f"[INFO] fixed classify_r=D.ESCAPE_R={D.ESCAPE_R:g}  vs.  "
              f"A.F.'ss determine_escape_radius(A)={principled_r:.6g}")
    except Exception as exc:
        print(f"[INFO] determine_escape_radius(A) failed (non-fatal): {exc}")

    feat_dim = int(td.X_grid.shape[-1])
    d_state = (feat_dim - 2) // 2
    alive_grid = td.meta.get("alive_mask_grid", None)
    have_alive = alive_grid is not None and bool(np.any(alive_grid))

    enc, dec, dmd, losses, loss_components, val_losses = trainer(
        td.X1, td.X2, latent_dim=latent_dim, epochs=epochs, batch_size=batch_size, lr=lr, device=device,
    )

    has_val = bool(np.any(np.isfinite(val_losses))) if len(val_losses) else False
    save_loss_curve(
        losses, dirs["res"] / "loss_curve.png", f"{name} Single Matrix Loss",
        extra_series=({"Validation": val_losses} if has_val else None),
        primary_label="Train" if has_val else None,
    )
    for key, title_suffix in (("rec", "Reconstruction"), ("lin", "Latent Linearity"), ("pred", "Decoded Prediction")):
        if key in loss_components:
            save_loss_curve(loss_components[key], dirs["res"] / f"loss_curve_{key}.png",
                             f"{name} Single Matrix Loss -- {title_suffix} Component")

    save_model(enc, Path(D.CHECKPOINT_DIR) / f"{name}_encoder_single_matrix.pth")
    save_model(dec, Path(D.CHECKPOINT_DIR) / f"{name}_decoder_single_matrix.pth")

    rho: Optional[float] = None
    A_op = getattr(dmd, "A", None)
    if A_op is not None:
        try:
            rho = float(torch.max(torch.abs(torch.linalg.eigvals(A_op.detach().cpu()))))
            print(f"{name.upper()} LATENT-OPERATOR SPECTRAL RADIUS:", rho)
        except Exception:
            pass

    Z_recon = reconstruct_true_final_snapshot(td, enc, dec, device)
    save_final_snapshot_image(Z_recon, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "recon_final_mask.png",
                               mode="mask", alive_mask=alive_grid)
    save_final_snapshot_image(Z_recon, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "recon_final_snapshot_mag.png",
                               mode="mag", alive_mask=alive_grid)

    k = int(D.PREDICT_EXTRA_STEPS)
    Z_pred = predict_next_snapshot(td, enc, dec, dmd, device, steps=k, escape_r=D.DYNAMICS_CLAMP_R)
    Z_true_next = iterate_true_next_snapshot(td, A, steps=k, escape_r=D.DYNAMICS_CLAMP_R)
    debug_final_state_stats(f"{name.upper()} SINGLE MATRIX (+{k})", Z_pred, D.ESCAPE_R)

    save_final_snapshot_image(Z_pred, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "pred_final_mask.png",
                               mode="mask", alive_mask=alive_grid)
    save_final_snapshot_image(Z_pred, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "pred_final_snapshot_mag.png",
                               mode="mag", alive_mask=alive_grid)
    save_final_snapshot_image(Z_true_next, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "true_next_final_mask.png",
                               mode="mask")
    save_final_snapshot_image(Z_true_next, escape_r=D.ESCAPE_R,
                               out_png=dirs["res"] / "true_next_final_snapshot_mag.png", mode="mag")

    iters_pred = teacher_forced_escape_iters(td, enc, dec, dmd, device, escape_r=D.ESCAPE_R)
    save_escape_image(iters_pred, max_iters=int(td.X_grid.shape[0]), out_png=dirs["res"] / "pred_escape_iters.png",
                       alive_mask=alive_grid)

    _save_fullgrid_comparison_images(name, td, enc, dec, dmd, device, dirs["res"])


    maxit = int(D.TRAIN_MAX_ITERS)
    rollout = predict_rollout_from_start_ae_dmd(td, enc, dec, dmd, device, steps=maxit, escape_r=D.DYNAMICS_CLAMP_R)

    Z_pred_final = rollout[-1]
    Z_true_final = td.X_grid[-1][..., :2 * d_state]

    save_final_snapshot_image(Z_pred_final, escape_r=D.ESCAPE_R,
                               out_png=dirs["res"] / "rollout_from_start_final_mask.png",
                               mode="mask", alive_mask=alive_grid)
    save_final_snapshot_image(Z_pred_final, escape_r=D.ESCAPE_R,
                               out_png=dirs["res"] / "rollout_from_start_final_mag.png",
                               mode="mag", alive_mask=alive_grid)

    rollout_final_m = next_step_prediction_metrics(Z_pred_final, Z_true_final)
    print_metric_block(f"{name.upper()} ROLLOUT x1 -> x{maxit} (MACRO, FULL GRID)", rollout_final_m)

    rollout_final_m_alive = None
    if have_alive:
        rollout_final_m_alive = next_step_prediction_metrics(Z_pred_final[alive_grid], Z_true_final[alive_grid])
        print_metric_block(f"{name.upper()} ROLLOUT x1 -> x{maxit} (MACRO, ALIVE-ONLY)", rollout_final_m_alive)

    n_check = min(int(getattr(D, "PREDICT_ROLLOUT_CHECK_STEPS", 10)), rollout.shape[0])
    rollout_rel_l2: list[float] = []
    rollout_rel_l2_alive: list[float] = []
    rollout_step_metrics: dict = {}
    for s in range(n_check):
        Z_pred_s = rollout[s]
        Z_true_s = td.X_grid[s + 1][..., :2 * d_state]
        m_s = next_step_prediction_metrics(Z_pred_s, Z_true_s)
        rollout_step_metrics[f"rollout_rel_l2_step_{s + 1:03d}"] = float(m_s["pred_rel_l2"])
        rollout_rel_l2.append(float(m_s["pred_rel_l2"]))
        if have_alive:
            m_s_alive = next_step_prediction_metrics(Z_pred_s[alive_grid], Z_true_s[alive_grid])
            rollout_step_metrics[f"rollout_rel_l2_alive_step_{s + 1:03d}"] = float(m_s_alive["pred_rel_l2"])
            rollout_rel_l2_alive.append(float(m_s_alive["pred_rel_l2"]))

    save_loss_curve(
        rollout_rel_l2, dirs["res"] / "rollout_rel_l2_vs_step.png",
        f"{name} Rollout Relative L2 Error vs Steps Beyond x1 (Full Grid)",
        xlabel="Steps beyond x1", ylabel="Relative L2 error", log_scale=False,
    )
    if rollout_rel_l2_alive:
        save_loss_curve(
            rollout_rel_l2_alive, dirs["res"] / "rollout_rel_l2_vs_step_alive.png",
            f"{name} Rollout Relative L2 Error vs Steps Beyond x1 (Alive-Only)",
            xlabel="Steps beyond x1", ylabel="Relative L2 error", log_scale=False,
        )

    ae_m = autoencoder_reconstruction_metrics(enc, dec, td.X, device)
    ae_m_alive = autoencoder_reconstruction_metrics_alive(enc, dec, td, device)
    dmd_m = dmd_one_step_metrics(enc, dec, dmd, td.X1, td.X2, device)
    pred_m = next_step_prediction_metrics(Z_pred, Z_true_next)
    print_metric_block(f"{name.upper()} AE (RECON, FULL GRID)", ae_m)
    print_metric_block(f"{name.upper()} AE (RECON, ALIVE-ONLY)", ae_m_alive)
    print_metric_block(f"{name.upper()} LATENT OPERATOR (ONE-STEP)", dmd_m)
    print_metric_block(f"{name.upper()} PREDICT (+{k} FROM TRUE xT, FULL GRID)", pred_m)

    pred_m_alive = None
    if have_alive:
        pred_m_alive = next_step_prediction_metrics(Z_pred[alive_grid], Z_true_next[alive_grid])
        print_metric_block(f"{name.upper()} PREDICT (+{k} FROM TRUE xT, ALIVE-ONLY)", pred_m_alive)

    metrics = {
        **ae_m, **ae_m_alive, **dmd_m, **pred_m,
        "predict_extra_steps": float(k),
        **({"dmd_spectral_radius": rho} if rho is not None else {}),
        **{f"rollout_final_{key}": val for key, val in rollout_final_m.items()},
        **rollout_step_metrics,
        "n_alive": float(td.meta.get("n_alive", -1)),
        "n_total": float(td.meta.get("n_total", -1)),
        **({f"rollout_final_alive_{key}": val for key, val in rollout_final_m_alive.items()}
           if rollout_final_m_alive is not None else {}),
        **({f"pred_alive_{key}": val for key, val in pred_m_alive.items()}
           if pred_m_alive is not None else {}),
    }
    metrics["rollout_final_anae_pct"] = float(OA.anae(
        torch.tensor(Z_true_final, dtype=torch.float32), torch.tensor(Z_pred_final, dtype=torch.float32),
    ))

    write_metrics_txt(dirs["res"] / "metrics.txt", metrics)
    return metrics, losses

def _load_rgb(path: Path) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def _resize_to_match(a: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if a.shape[:2] == tuple(shape):
        return a
    from PIL import Image
    im = Image.fromarray(a.astype(np.uint8)).resize((shape[1], shape[0]), resample=Image.NEAREST)
    return np.asarray(im, dtype=a.dtype if a.dtype == bool else np.float32)


def _gray_and_excluded(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    excluded = (r != g) | (g != b)
    gray = r
    return gray, excluded


def _otsu_mask(a: np.ndarray, valid: Optional[np.ndarray] = None) -> np.ndarray:
    sample = a if valid is None else a[valid]
    if sample.size == 0:
        return np.zeros_like(a, dtype=bool)
    if float(sample.max()) <= float(sample.min()) + 1e-9:
        return a >= 127.5

    hist, _ = np.histogram(sample, bins=256, range=(0, 255))
    total = int(sample.size)
    sum_all = float(np.dot(hist, np.arange(256)))
    sum_b = w_b = 0.0
    best_thr, best_var = 128, -1.0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > best_var:
            best_var, best_thr = var_between, t
    return a > best_thr


def compare_image_to_reference(obtained_path: Path, reference_path: Path) -> dict:
    a_rgb = _load_rgb(obtained_path)
    b_rgb = _load_rgb(reference_path)

    a_gray, a_excluded = _gray_and_excluded(a_rgb)
    b_gray, _ = _gray_and_excluded(b_rgb)

    if a_gray.shape != b_gray.shape:
        a_gray = _resize_to_match(a_gray, b_gray.shape)
        a_excluded = _resize_to_match(a_excluded, b_gray.shape)

    valid = ~a_excluded
    n_excluded = int(a_excluded.sum())
    if not np.any(valid):
        return {"image_mse": float("nan"), "image_ncc": float("nan"),
                "image_iou": float("nan"), "image_dice": float("nan"), "image_excluded_px": n_excluded}

    a_v, b_v = a_gray[valid], b_gray[valid]
    diff = a_v - b_v
    mse = float(np.mean(diff * diff))

    a0, b0 = a_v - a_v.mean(), b_v - b_v.mean()
    denom = float(np.sqrt(np.sum(a0 * a0) * np.sum(b0 * b0)))
    ncc = float(np.sum(a0 * b0) / denom) if denom > 1e-12 else (1.0 if np.allclose(a_v, b_v) else 0.0)

    mask_a = _otsu_mask(a_gray, valid)
    mask_b = _otsu_mask(b_gray, valid)
    mask_a_v, mask_b_v = mask_a[valid], mask_b[valid]
    inter = float(np.logical_and(mask_a_v, mask_b_v).sum())
    union = float(np.logical_or(mask_a_v, mask_b_v).sum())
    iou = inter / union if union > 0 else 1.0
    denom_dice = float(mask_a_v.sum() + mask_b_v.sum())
    dice = (2 * inter) / denom_dice if denom_dice > 0 else 1.0

    return {"image_mse": mse, "image_ncc": ncc, "image_iou": iou, "image_dice": dice, "image_excluded_px": n_excluded}


def _make_contact_sheet(images: dict[str, Path], out_png: Path, *, title: str, tile_size: int = 220) -> str:
    from PIL import Image, ImageDraw

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    names = list(images.keys())
    label_h = 22
    canvas = Image.new("RGB", (tile_size * max(len(names), 1), tile_size + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), title, fill="black")

    for i, name in enumerate(names):
        im = Image.open(images[name]).convert("RGB").resize((tile_size, tile_size), Image.NEAREST)
        canvas.paste(im, (i * tile_size, label_h))
        draw.rectangle([i * tile_size, label_h, i * tile_size + tile_size - 1, label_h + tile_size - 1], outline="gray")
        draw.text((i * tile_size + 4, label_h + 4), name, fill=(0, 120, 255))

    canvas.save(out_png)
    return str(out_png)


def _write_image_comparison_csv(rows: list[dict], out_csv: Path) -> str:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["architecture", "image_kind", "image_mse", "image_ncc", "image_iou", "image_dice", "image_excluded_px"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})
    return str(out_csv)

def _save_multi_curve(
    curves: dict[str, list[float]], out_png: Path, *, title: str, xlabel: str, ylabel: str, log_scale: bool = False,
) -> str:
    import matplotlib.pyplot as plt

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plot_fn = plt.semilogy if log_scale else plt.plot
    for name, ys in curves.items():
        if not ys:
            continue
        ys_arr = np.asarray(ys, dtype=np.float32)
        xs = np.arange(1, len(ys_arr) + 1)
        if log_scale:
            ys_arr = np.where(ys_arr > 0, ys_arr, np.nan)
        plot_fn(xs, ys_arr, label=name, marker="o", markersize=3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    plt.savefig(out_png, dpi=200)
    plt.close()
    return str(out_png)


def _save_grouped_bars(values: dict[str, float], out_png: Path, *, title: str, ylabel: str) -> str:
    import matplotlib.pyplot as plt

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    names = list(values.keys())
    ys = [values[n] for n in names]

    plt.figure()
    plt.bar(names, ys)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    plt.savefig(out_png, dpi=200)
    plt.close()
    return str(out_png)


def _write_metrics_comparison_csv(rows: dict[str, dict], out_csv: Path) -> str:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    all_keys: list[str] = []
    seen: set[str] = set()
    for m in rows.values():
        for k in m.keys():
            if k not in seen:
                seen.add(k)
                all_keys.append(k)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["architecture", *all_keys])
        for name, m in rows.items():
            w.writerow([name, *[m.get(k, "") for k in all_keys]])
    return str(out_csv)

def run_comparison(
    device: Optional[torch.device] = None,
    *,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    latent_dim: Optional[int] = None,
    architectures: Optional[list[str]] = None,
) -> dict:
    if device is None:
        device = pick_device()
    print("DEVICE:", device)

    epochs = int(D.AE_EPOCHS if epochs is None else epochs)
    batch_size = int(D.AE_BATCH_SIZE if batch_size is None else batch_size)
    lr = float(D.AE_LR if lr is None else lr)
    latent_dim = int(D.LATENT_DIM if latent_dim is None else latent_dim)

    set_recommended_matplotlib()

    out_root = Path("out") / COMPARISON_DIRNAME
    img_cmp_dir = out_root / "images"
    out_root.mkdir(parents=True, exist_ok=True)
    img_cmp_dir.mkdir(parents=True, exist_ok=True)

    print("\n================ ARCHITECTURE COMPARISON ================\n")
    print(f"other-architecture settings: latent_dim={latent_dim} epochs={epochs} "
          f"batch_size={batch_size} lr={lr:g}  (see docstring: 'mine' always follows defines.py)\n")

    other_names = [n for n in OA.ARCHITECTURES if architectures is None or n in architectures]

    all_metrics: dict[str, dict] = {}
    all_losses: dict[str, list[float]] = {}

    all_metrics["mine"] = run_mine_experiment(device)

    td = build_matrix_c_grid_training_data(
        data_dir=D.A_DATA_DIR, source=D.SINGLE_MATRIX_SOURCE, index=D.SINGLE_MATRIX_INDEX,
        c_re_min=D.C_RE_MIN, c_re_max=D.C_RE_MAX, c_im_min=D.C_IM_MIN, c_im_max=D.C_IM_MAX,
        c_re_n=D.SINGLE_MATRIX_C_RE_N, c_im_n=D.SINGLE_MATRIX_C_IM_N,
        max_iters=D.TRAIN_MAX_ITERS, escape_r=D.DYNAMICS_CLAMP_R, classify_r=D.ESCAPE_R,
        filter_escaped=D.FILTER_ESCAPED_FOR_TRAINING, keep_escaped_fraction=D.KEEP_ESCAPED_FRACTION,
    )
    A = load_one_A_matrix(D.A_DATA_DIR, source=D.SINGLE_MATRIX_SOURCE, index=D.SINGLE_MATRIX_INDEX)

    mine_enc, mine_dec, mine_dmd = reload_mine_model(td, device)
    _save_fullgrid_comparison_images("mine", td, mine_enc, mine_dec, mine_dmd, device, _arch_dirs("mine")["res"])

    for name in other_names:
        trainer = OA.ARCHITECTURES[name]
        metrics, losses = run_full_experiment_for_architecture(
            name, trainer, td, A, device, latent_dim=latent_dim, epochs=epochs, batch_size=batch_size, lr=lr,
        )
        all_metrics[name] = metrics
        all_losses[name] = losses
        print_metric_block(f"{name.upper()} SUMMARY", metrics)

    _write_metrics_comparison_csv(all_metrics, out_root / "metrics_comparison.csv")

    all_curves = {n: _extract_rollout_curve(m, alive=False) for n, m in all_metrics.items()}
    all_curves = {n: c for n, c in all_curves.items() if c}
    all_curves_alive = {n: _extract_rollout_curve(m, alive=True) for n, m in all_metrics.items()}
    all_curves_alive = {n: c for n, c in all_curves_alive.items() if c}

    _save_multi_curve(
        all_curves, out_root / "rollout_rel_l2_vs_step.png",
        title="Rollout Relative L2 Error vs Steps Beyond x1 (Full Grid)",
        xlabel="Steps beyond x1", ylabel="Relative L2 error",
    )
    if all_curves_alive:
        _save_multi_curve(
            all_curves_alive, out_root / "rollout_rel_l2_vs_step_alive.png",
            title="Rollout Relative L2 Error vs Steps Beyond x1 (Alive-Only)",
            xlabel="Steps beyond x1", ylabel="Relative L2 error",
        )
    if all_losses:
        _save_multi_curve(
            all_losses, out_root / "train_loss_curves.png",
            title="Training Loss (mine excluded -- see run_mine_experiment, "
                  "its own curve is at out/single-matrix/results/loss_curve.png)",
            xlabel="Epoch", ylabel="Loss", log_scale=True,
        )
    _save_grouped_bars(
        {n: m.get("rollout_final_pred_rel_l2", float("nan")) for n, m in all_metrics.items()},
        out_root / "bar_rollout_final_rel_l2.png",
        title=f"Rollout x1 -> x{int(D.TRAIN_MAX_ITERS)} Relative L2 Error (lower is better)",
        ylabel="Relative L2 error",
    )
    _save_grouped_bars(
        {n: m.get("ae_alive_rel_l2", float("nan")) for n, m in all_metrics.items()},
        out_root / "bar_ae_reconstruction_rel_l2.png",
        title="Autoencoder Reconstruction Relative L2 Error, alive rows only (lower is better)",
        ylabel="Relative L2 error",
    )

    print("\n================ IMAGE COMPARISON ================\n")
    gt_dirs = _arch_dirs("mine")
    image_kinds = {
        "reconstruction_mask": ("recon_final_mask_fullgrid.png", "gt_final_mask.png"),
        "rollout_final_mask": ("rollout_from_start_final_mask_fullgrid.png", "gt_final_mask.png"),
        "escape_iters": ("pred_escape_iters_fullgrid.png", "gt_escape_iters.png"),
    }
    all_names = ["mine", *other_names]
    image_rows: list[dict] = []

    for kind_label, (obtained_name, reference_name) in image_kinds.items():
        reference_path = gt_dirs["td"] / reference_name
        if not reference_path.exists():
            print(f"  [WARN] skipping image kind {kind_label!r}: reference {reference_path} not found")
            continue

        sheet_paths = {"ground_truth": reference_path}
        for name in all_names:
            obtained_path = _arch_dirs(name)["res"] / obtained_name
            if not obtained_path.exists():
                print(f"  [WARN] skipping {name}/{obtained_name}: not found")
                continue
            sheet_paths[name] = obtained_path
            try:
                cmp = compare_image_to_reference(obtained_path, reference_path)
            except Exception as exc:
                print(f"  [WARN] image comparison failed for {name}/{obtained_name}: {exc}")
                continue
            print(f"  {name:12s} | {kind_label:22s} | mse={cmp['image_mse']:.2f} ncc={cmp['image_ncc']:.4f} "
                  f"iou={cmp['image_iou']:.4f} dice={cmp['image_dice']:.4f} excluded_px={cmp['image_excluded_px']}")
            image_rows.append({"architecture": name, "image_kind": kind_label, **cmp})

        if len(sheet_paths) > 1:
            _make_contact_sheet(
                sheet_paths, img_cmp_dir / f"contact_sheet_{obtained_name.replace('.png', '')}.png",
                title=kind_label,
            )

    _write_image_comparison_csv(image_rows, out_root / "image_comparison.csv")

    ranked = sorted(all_metrics.items(), key=lambda kv: kv[1].get("rollout_final_pred_rel_l2", float("inf")))
    print(f"\n================ LEADERBOARD: x1 -> x{int(D.TRAIN_MAX_ITERS)} ROLLOUT (lower is better) ================")
    for rank, (name, m) in enumerate(ranked, start=1):
        anae_val = m.get("rollout_final_anae_pct")
        anae_str = f"{anae_val:.2f}%" if anae_val is not None else "n/a"
        print(f"  {rank}. {name:12s}  rel_l2={m.get('rollout_final_pred_rel_l2', float('nan')):.4f}   ANAE={anae_str}")

    print(f"\nFull comparison (metrics CSV, image CSV, contact sheets, plots) written to: {out_root}")
    print(f"Per-architecture experiment folders: out/single-matrix/ (mine), " +
          ", ".join(f"out/{n}/single-matrix/" for n in other_names))

    return all_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare my architecture against reference architectures")
    parser.add_argument("--epochs", type=int, default=None, help="override defines.AE_EPOCHS (other architectures only)")
    parser.add_argument("--batch-size", type=int, default=None, help="override defines.AE_BATCH_SIZE (other architectures only)")
    parser.add_argument("--lr", type=float, default=None, help="override defines.AE_LR (other architectures only)")
    parser.add_argument("--latent-dim", type=int, default=None, help="override defines.LATENT_DIM (other architectures only)")
    parser.add_argument(
        "--architectures", type=str, default=None,
        help="comma-separated subset of {lusch,dlkoopman,dldmd} to run alongside 'mine' (default: all)",
    )
    args = parser.parse_args()

    run_comparison(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, latent_dim=args.latent_dim,
        architectures=(args.architectures.split(",") if args.architectures else None),
    )