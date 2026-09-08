from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import defines as D
from data_loader import load_all_A_matrices, split_explicit_matrix_indices
from prepare_training_data import build_matrix_c_grid_training_data_many_matrices
from train_autoencoder import train_autoencoder
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
    predict_rollout_from_start_ae_dmd,
    next_step_prediction_metrics,
    teacher_forced_escape_iters,
)
from mandelbrot_reconstruct import save_final_snapshot_image, save_escape_image
from experiment_common import (
    pick_device,
    fit_streamed_dmd_from_td_list,
    print_metric_block,
    mean_metric_dict,
    write_metrics_txt,
)
import other_architectures as OA


MULTI_COMPARISON_DIRNAME = "aa__comparison_multi_output__aa"

def _arch_multi_dirs(name: str) -> dict:
    root = Path("out") / "multi-matrix" if name == "mine" else Path("out") / name / "multi-matrix"
    return {"root": root, "td": root / "training-data", "res": root / "results"}


def _make_arch_multi_dirs(name: str) -> dict:
    dirs = _arch_multi_dirs(name)
    dirs["td"].mkdir(parents=True, exist_ok=True)
    dirs["res"].mkdir(parents=True, exist_ok=True)
    return dirs


def _read_metrics_txt(path: Path) -> dict:
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


def _extract_rollout_curve(metrics: dict, *, alive: bool = False) -> list:
    prefix = "rollout_rel_l2_alive_step_" if alive else "rollout_rel_l2_step_"
    pairs = sorted(
        (int(k[len(prefix):]), v) for k, v in metrics.items()
        if k.startswith(prefix) and k[len(prefix):].isdigit()
    )
    return [v for _, v in pairs]

def _eval_model_on_test_list(
    enc, dec, dmd, td_test_list, test_idx, A_all, device, *,
    save_example_dir: Optional[Path] = None, label: str = "",
) -> dict:
    k = int(D.PREDICT_EXTRA_STEPS)
    maxit = int(D.TRAIN_MAX_ITERS)
    n_check = min(int(getattr(D, "PREDICT_ROLLOUT_CHECK_STEPS", 10)), max(maxit - 1, 0))

    ae_all, ae_alive_all, dmd_all = [], [], []
    pred_all, pred_alive_all = [], []
    rollout_final_all, rollout_final_alive_all = [], []
    step_curves_all, step_curves_alive_all = [], []

    for j, td_test in enumerate(td_test_list):
        A_test = A_all[int(test_idx[j])]
        alive_grid = td_test.meta.get("alive_mask_grid", None)
        have_alive = alive_grid is not None and bool(np.any(alive_grid))

        ae_m = autoencoder_reconstruction_metrics(enc, dec, td_test.X, device)
        ae_m_alive = autoencoder_reconstruction_metrics_alive(enc, dec, td_test, device)
        dmd_m = dmd_one_step_metrics(enc, dec, dmd, td_test.X1, td_test.X2, device)

        Z_pred = predict_next_snapshot(td_test, enc, dec, dmd, device, steps=k, escape_r=D.DYNAMICS_CLAMP_R)
        Z_true_next = iterate_true_next_snapshot(td_test, A_test, steps=k, escape_r=D.DYNAMICS_CLAMP_R)
        pred_m = next_step_prediction_metrics(Z_pred, Z_true_next)
        pred_m_alive = (
            next_step_prediction_metrics(Z_pred[alive_grid], Z_true_next[alive_grid]) if have_alive else None
        )

        rollout = predict_rollout_from_start_ae_dmd(
            td_test, enc, dec, dmd, device, steps=maxit, escape_r=D.DYNAMICS_CLAMP_R,
        )
        d_state = int((int(td_test.X_grid.shape[-1]) - 2) // 2)
        Z_pred_final = rollout[-1]
        Z_true_final = td_test.X_grid[-1][..., :2 * d_state]
        final_m = next_step_prediction_metrics(Z_pred_final, Z_true_final)
        rollout_final_all.append(final_m)
        if have_alive:
            rollout_final_alive_all.append(
                next_step_prediction_metrics(Z_pred_final[alive_grid], Z_true_final[alive_grid])
            )

        step_curve, step_curve_alive = [], []
        for s in range(n_check):
            Z_pred_s = rollout[s]
            Z_true_s = td_test.X_grid[s + 1][..., :2 * d_state]
            step_curve.append(float(next_step_prediction_metrics(Z_pred_s, Z_true_s)["pred_rel_l2"]))
            if have_alive:
                step_curve_alive.append(
                    float(next_step_prediction_metrics(Z_pred_s[alive_grid], Z_true_s[alive_grid])["pred_rel_l2"])
                )
        step_curves_all.append(step_curve)
        if step_curve_alive:
            step_curves_alive_all.append(step_curve_alive)

        ae_all.append(ae_m)
        ae_alive_all.append(ae_m_alive)
        dmd_all.append(dmd_m)
        pred_all.append(pred_m)
        if pred_m_alive is not None:
            pred_alive_all.append(pred_m_alive)

        print_metric_block(f"[{label}] TEST MATRIX {int(test_idx[j])} DMD (ONE-STEP)", dmd_m)
        print_metric_block(f"[{label}] TEST MATRIX {int(test_idx[j])} ROLLOUT x1 -> x{maxit} (MACRO, FULL GRID)", final_m)

        if save_example_dir is not None and j == 0:
            save_ground_truth_final_mask(td_test, D.ESCAPE_R, save_example_dir / "test_gt_final_mask.png", scale=D.IMAGE_SCALE)
            save_ground_truth_escape_iters(td_test, D.ESCAPE_R, save_example_dir / "test_gt_escape_iters.png")
            Z_recon = reconstruct_true_final_snapshot(td_test, enc, dec, device)
            save_final_snapshot_image(Z_recon, escape_r=D.ESCAPE_R, out_png=save_example_dir / "test_recon_final_mask.png",
                                      mode="mask", alive_mask=alive_grid)
            save_final_snapshot_image(Z_pred_final, escape_r=D.ESCAPE_R,
                                      out_png=save_example_dir / "test_rollout_from_start_final_mask.png",
                                      mode="mask", alive_mask=alive_grid)
            iters_pred = teacher_forced_escape_iters(td_test, enc, dec, dmd, device, escape_r=D.ESCAPE_R)
            save_escape_image(iters_pred, max_iters=int(td_test.X_grid.shape[0]),
                              out_png=save_example_dir / "test_pred_escape_iters.png", alive_mask=alive_grid)

    out: dict[str, float] = {}
    out.update(mean_metric_dict(ae_all))
    out.update(mean_metric_dict(ae_alive_all))
    out.update(mean_metric_dict(dmd_all))
    out.update(mean_metric_dict(pred_all))
    if pred_alive_all:
        out.update({f"pred_alive_{kk}": vv for kk, vv in mean_metric_dict(pred_alive_all).items()})
    out["predict_extra_steps"] = float(k)

    out.update({f"rollout_final_{kk}": vv for kk, vv in mean_metric_dict(rollout_final_all).items()})
    if rollout_final_alive_all:
        out.update({f"rollout_final_alive_{kk}": vv for kk, vv in mean_metric_dict(rollout_final_alive_all).items()})

    if step_curves_all:
        mean_curve = np.mean(np.asarray(step_curves_all, dtype=np.float64), axis=0).tolist()
        for s, val in enumerate(mean_curve):
            out[f"rollout_rel_l2_step_{s + 1:03d}"] = float(val)
    if step_curves_alive_all:
        mean_curve_alive = np.mean(np.asarray(step_curves_alive_all, dtype=np.float64), axis=0).tolist()
        for s, val in enumerate(mean_curve_alive):
            out[f"rollout_rel_l2_alive_step_{s + 1:03d}"] = float(val)

    A_op = getattr(dmd, "A", None)
    if A_op is not None:
        try:
            out["dmd_spectral_radius"] = float(torch.max(torch.abs(torch.linalg.eigvals(A_op.detach().cpu()))))
        except Exception:
            pass
    return out


def _train_mine_on_shared(td_train_list, device):
    enc, dec, losses, loss_components, val_losses = train_autoencoder(
        [td.X1 for td in td_train_list],
        [td.X2 for td in td_train_list],
        latent_dim=D.LATENT_DIM, epochs=D.AE_EPOCHS, batch_size=D.AE_BATCH_SIZE, lr=D.AE_LR, device=device,
    )
    dmd = fit_streamed_dmd_from_td_list(enc, td_train_list, device)
    return enc, dec, dmd

def _write_comparison_csv(all_metrics: dict, out_csv: Path) -> None:
    methods = list(all_metrics.keys())
    keys = sorted({k for m in all_metrics.values() for k in m.keys()})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", *methods])
        for key in keys:
            row = [key]
            for m in methods:
                v = all_metrics[m].get(key, None)
                row.append("" if v is None else f"{v:.10e}")
            w.writerow(row)
    print(f"  wrote {out_csv}")


def _save_rollout_overlay(all_metrics: dict, out_png: Path) -> None:
    curves = {name: _extract_rollout_curve(m, alive=True) for name, m in all_metrics.items()}
    curves = {name: c for name, c in curves.items() if c}
    if not curves:
        print("[overlay] no alive rollout curves to plot")
        return
    names = list(curves.keys())
    save_loss_curve(
        curves[names[0]], out_png,
        "Multi-Matrix Held-Out Rollout Relative L2 (alive) vs Steps Beyond x1",
        xlabel="Steps beyond x1", ylabel="Relative L2 error (alive)",
        log_scale=False,
        extra_series={n: curves[n] for n in names[1:]},
        primary_label=names[0],
    )
    print(f"  wrote {out_png}")

def run_comparison_multi(
    device: Optional[torch.device] = None, *,
    epochs: Optional[int] = None, batch_size: Optional[int] = None,
    lr: Optional[float] = None, latent_dim: Optional[int] = None,
    architectures: Optional[list] = None, reuse_mine: bool = True,
) -> dict:
    if device is None:
        device = pick_device()
    print("DEVICE:", device)

    epochs = int(D.AE_EPOCHS if epochs is None else epochs)
    batch_size = int(D.AE_BATCH_SIZE if batch_size is None else batch_size)
    lr = float(D.AE_LR if lr is None else lr)
    latent_dim = int(D.LATENT_DIM if latent_dim is None else latent_dim)

    out_root = Path("out") / MULTI_COMPARISON_DIRNAME
    out_root.mkdir(parents=True, exist_ok=True)

    print("\n============== MULTI-MATRIX ARCHITECTURE COMPARISON ==============\n")

    A_all = load_all_A_matrices(D.A_DATA_DIR, source=D.MULTI_MATRIX_SOURCE)
    total = int(A_all.shape[0])
    train_idx, test_idx = split_explicit_matrix_indices(
        total_count=total, train_count=D.MULTI_MATRIX_TRAIN_COUNT,
        test_count=D.MULTI_MATRIX_TEST_COUNT, seed=D.MULTI_MATRIX_SPLIT_SEED,
    )
    print("TRAIN IDX:", train_idx.tolist(), " TEST IDX:", test_idx.tolist())

    common_kwargs = dict(
        data_dir=D.A_DATA_DIR, source=D.MULTI_MATRIX_SOURCE,
        c_re_min=D.C_RE_MIN, c_re_max=D.C_RE_MAX, c_im_min=D.C_IM_MIN, c_im_max=D.C_IM_MAX,
        c_re_n=D.MULTI_MATRIX_C_RE_N, c_im_n=D.MULTI_MATRIX_C_IM_N,
        max_iters=D.TRAIN_MAX_ITERS, escape_r=D.DYNAMICS_CLAMP_R,
        classify_r=D.ESCAPE_R, filter_escaped=D.FILTER_ESCAPED_FOR_TRAINING,
    )
    td_train_list = build_matrix_c_grid_training_data_many_matrices(indices=train_idx, **common_kwargs)
    for td in td_train_list:
        td.X_grid = None
    td_test_list = build_matrix_c_grid_training_data_many_matrices(indices=test_idx, **common_kwargs)

    all_metrics: dict[str, dict] = {}

    mine_file = _arch_multi_dirs("mine")["res"] / "mean_test_metrics.txt"
    if reuse_mine and mine_file.exists():
        print(f"[mine] reusing existing {mine_file} (from `python main.py multi-matrix`)")
        all_metrics["mine"] = _read_metrics_txt(mine_file)
    else:
        print("[mine] no existing multi-matrix run found -- training the proposed pipeline on the shared split")
        enc, dec, dmd = _train_mine_on_shared(td_train_list, device)
        dirs = _make_arch_multi_dirs("mine")
        all_metrics["mine"] = _eval_model_on_test_list(
            enc, dec, dmd, td_test_list, test_idx, A_all, device,
            save_example_dir=dirs["res"], label="mine",
        )
        write_metrics_txt(dirs["res"] / "mean_test_metrics.txt", all_metrics["mine"])

    X1_pool = np.concatenate([np.asarray(td.X1, dtype=np.float32) for td in td_train_list], axis=0)
    X2_pool = np.concatenate([np.asarray(td.X2, dtype=np.float32) for td in td_train_list], axis=0)
    for td in td_train_list:
        td.X1 = None
        td.X2 = None
    print(f"[baselines] pooled train rows: {X1_pool.shape[0]}  feat_dim: {X1_pool.shape[1]}")

    names = [n for n in OA.ARCHITECTURES if architectures is None or n in architectures]
    for name in names:
        print(f"\n---------------- TRAINING (multi): {name} ----------------")
        if name in OA.ARCHITECTURE_CITATIONS:
            print(f"  ({OA.ARCHITECTURE_CITATIONS[name]})")
        trainer = OA.ARCHITECTURES[name]
        enc, dec, dmd, losses, loss_components, val_losses = trainer(
            X1_pool, X2_pool, latent_dim=latent_dim, epochs=epochs, batch_size=batch_size, lr=lr, device=device,
        )
        dirs = _make_arch_multi_dirs(name)
        save_loss_curve(losses, dirs["res"] / "loss_curve.png", f"{name} Multi-Matrix Loss")
        m = _eval_model_on_test_list(
            enc, dec, dmd, td_test_list, test_idx, A_all, device,
            save_example_dir=dirs["res"], label=name,
        )
        write_metrics_txt(dirs["res"] / "mean_test_metrics.txt", m)
        all_metrics[name] = m
        print_metric_block(f"{name.upper()} MULTI HELD-OUT MEAN", m)

    _write_comparison_csv(all_metrics, out_root / "metrics_comparison_multi.csv")
    _save_rollout_overlay(all_metrics, out_root / "rollout_rel_l2_vs_step_alive_multi.png")
    print(f"\nMulti-matrix comparison written under {out_root}")
    return all_metrics


if __name__ == "__main__":
    run_comparison_multi()