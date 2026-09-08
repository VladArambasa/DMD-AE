from __future__ import annotations

import os
import numpy as np
import torch

import defines as D
from utils import save_model, to_tensor
from data_loader import load_one_A_matrix
from prepare_training_data import build_matrix_c_grid_training_data, save_training_npz, determine_escape_radius
from train_autoencoder import train_autoencoder
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
from experiment_common import (
    pick_device,
    make_out_dirs,
    print_metric_block,
    write_metrics_txt,
    debug_final_state_stats,
)

def run_single_matrix(device: torch.device | None = None) -> None:
    if device is None:
        device = pick_device()

    print("DEVICE:", device)
    print("CWD:", os.getcwd())

    dirs = make_out_dirs("single-matrix")

    print("\n================ SINGLE MATRIX CHECK ================\n")


    A = load_one_A_matrix(D.A_DATA_DIR, source=D.SINGLE_MATRIX_SOURCE, index=D.SINGLE_MATRIX_INDEX)

    try:
        principled_r = determine_escape_radius(A)
    except Exception as exc:
        print(f"[INFO] determine_escape_radius(A) failed (non-fatal): {exc}")

    td = build_matrix_c_grid_training_data(
        data_dir=D.A_DATA_DIR,
        source=D.SINGLE_MATRIX_SOURCE,
        index=D.SINGLE_MATRIX_INDEX,
        c_re_min=D.C_RE_MIN,
        c_re_max=D.C_RE_MAX,
        c_im_min=D.C_IM_MIN,
        c_im_max=D.C_IM_MAX,
        c_re_n=D.SINGLE_MATRIX_C_RE_N,
        c_im_n=D.SINGLE_MATRIX_C_IM_N,
        max_iters=D.TRAIN_MAX_ITERS,
        escape_r=D.DYNAMICS_CLAMP_R,
        classify_r=D.ESCAPE_R,
        filter_escaped=D.FILTER_ESCAPED_FOR_TRAINING,
        keep_escaped_fraction=D.KEEP_ESCAPED_FRACTION,
    )


    save_training_npz(dirs["td"] / "training_single_matrix.npz", td)
    save_ground_truth_escape_iters(td, D.ESCAPE_R, dirs["td"] / "gt_escape_iters.png")
    save_ground_truth_final_mask(td, D.ESCAPE_R, dirs["td"] / "gt_final_mask.png", scale=D.IMAGE_SCALE)

    feat_dim = int(td.X_grid.shape[-1])
    d = (feat_dim - 2) // 2

    alive_grid = td.meta.get("alive_mask_grid", None)


    enc, dec, losses, loss_components, val_losses = train_autoencoder(
        td.X1,
        td.X2,
        latent_dim=D.LATENT_DIM,
        epochs=D.AE_EPOCHS,
        batch_size=D.AE_BATCH_SIZE,
        lr=D.AE_LR,
        device=device,
    )

    has_val = bool(np.any(np.isfinite(val_losses))) if len(val_losses) else False
    save_loss_curve(
        losses, dirs["res"] / "loss_curve.png", "Single Matrix AE Loss",
        extra_series=({"Validation": val_losses} if has_val else None),
        primary_label="Train" if has_val else None,
    )
    save_loss_curve(loss_components["rec"], dirs["res"] / "loss_curve_rec.png",
                     "Single Matrix AE Loss -- Reconstruction Component")
    save_loss_curve(loss_components["lin"], dirs["res"] / "loss_curve_lin.png",
                     "Single Matrix AE Loss -- Latent Linearity Component")
    save_loss_curve(loss_components["pred"], dirs["res"] / "loss_curve_pred.png",
                     "Single Matrix AE Loss -- Decoded Prediction Component")
    save_model(enc, os.path.join(D.CHECKPOINT_DIR, "encoder_single_matrix.pth"))
    save_model(dec, os.path.join(D.CHECKPOINT_DIR, "decoder_single_matrix.pth"))

    with torch.no_grad():
        Z1 = enc(to_tensor(td.X1, device)).detach().cpu().numpy()
        Z2 = enc(to_tensor(td.X2, device)).detach().cpu().numpy()



    dmd = fit_dmd_on_arrays(Z1, Z2, device=device, ridge=D.DMD_RIDGE)
    rho = float(np.max(np.abs(np.linalg.eigvals(dmd.A.detach().cpu().numpy()))))
    print("SINGLE DMD SPECTRAL RADIUS:", rho)


    Z_recon = reconstruct_true_final_snapshot(td, enc, dec, device)
    save_final_snapshot_image(Z_recon, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "recon_final_mask.png",
                               mode="mask", alive_mask=alive_grid)
    save_final_snapshot_image(Z_recon, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "recon_final_snapshot_mag.png",
                               mode="mag", alive_mask=alive_grid)


    k = int(D.PREDICT_EXTRA_STEPS)

    Z_pred = predict_next_snapshot(td, enc, dec, dmd, device, steps=k, escape_r=D.DYNAMICS_CLAMP_R)
    Z_true_next = iterate_true_next_snapshot(td, A, steps=k, escape_r=D.DYNAMICS_CLAMP_R)
    debug_final_state_stats(f"SINGLE MATRIX (+{k})", Z_pred, D.ESCAPE_R)

    save_final_snapshot_image(Z_pred, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "pred_final_mask.png",
                               mode="mask", alive_mask=alive_grid)
    save_final_snapshot_image(Z_pred, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "pred_final_snapshot_mag.png",
                               mode="mag", alive_mask=alive_grid)
    save_final_snapshot_image(Z_true_next, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "true_next_final_mask.png",
                               mode="mask")
    save_final_snapshot_image(Z_true_next, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "true_next_final_snapshot_mag.png",
                               mode="mag")


    iters_pred = teacher_forced_escape_iters(td, enc, dec, dmd, device, escape_r=D.ESCAPE_R)
    save_escape_image(iters_pred, max_iters=int(td.X_grid.shape[0]), out_png=dirs["res"] / "pred_escape_iters.png",
                       alive_mask=alive_grid)


    maxit = int(D.TRAIN_MAX_ITERS)
    rollout = predict_rollout_from_start_ae_dmd(
        td, enc, dec, dmd, device, steps=maxit, escape_r=D.DYNAMICS_CLAMP_R,
    )


    d_state = int((int(td.X_grid.shape[-1]) - 2) // 2)

    Z_pred_final = rollout[-1]
    Z_true_final = td.X_grid[-1][..., :2 * d_state]
    save_final_snapshot_image(Z_pred_final, escape_r=D.ESCAPE_R,
                              out_png=dirs["res"] / "rollout_from_start_final_mask.png",
                              mode="mask", alive_mask=alive_grid)
    save_final_snapshot_image(Z_pred_final, escape_r=D.ESCAPE_R,
                              out_png=dirs["res"] / "rollout_from_start_final_mag.png", mode="mag",
                              alive_mask=alive_grid)

    rollout_final_m = next_step_prediction_metrics(Z_pred_final, Z_true_final)
    print_metric_block(f"SINGLE MATRIX AE+DMD ROLLOUT x1 -> x{maxit} (MACRO, FULL GRID)", rollout_final_m)


    rollout_final_m_alive = None
    if alive_grid is not None and bool(np.any(alive_grid)):
        rollout_final_m_alive = next_step_prediction_metrics(
            Z_pred_final[alive_grid], Z_true_final[alive_grid],
        )
        print_metric_block(
            f"SINGLE MATRIX AE+DMD ROLLOUT x1 -> x{maxit} (MACRO, ALIVE-ONLY, "
            f"{int(np.count_nonzero(alive_grid))}/{alive_grid.size} px)",
            rollout_final_m_alive,
        )


    n_check = min(int(getattr(D, "PREDICT_ROLLOUT_CHECK_STEPS", 10)), rollout.shape[0])
    rollout_rel_l2: list[float] = []
    rollout_rel_l2_alive: list[float] = []
    rollout_step_metrics: dict = {}

    for s in range(n_check):
        true_iter = s + 2
        Z_pred_s = rollout[s]
        Z_true_s = td.X_grid[s + 1][..., :2 * d_state]

        m_s = next_step_prediction_metrics(Z_pred_s, Z_true_s)
        print_metric_block(f"SINGLE MATRIX AE+DMD ROLLOUT, {s + 1} STEP(S) IN (x{true_iter}, FULL GRID)", m_s)

        rollout_step_metrics[f"rollout_rel_l2_step_{s + 1:03d}"] = float(m_s["pred_rel_l2"])
        rollout_rel_l2.append(float(m_s["pred_rel_l2"]))

        if alive_grid is not None and bool(np.any(alive_grid)):
            m_s_alive = next_step_prediction_metrics(Z_pred_s[alive_grid], Z_true_s[alive_grid])
            print_metric_block(
                f"SINGLE MATRIX AE+DMD ROLLOUT, {s + 1} STEP(S) IN (x{true_iter}, ALIVE-ONLY)", m_s_alive,
            )
            rollout_step_metrics[f"rollout_rel_l2_alive_step_{s + 1:03d}"] = float(m_s_alive["pred_rel_l2"])
            rollout_rel_l2_alive.append(float(m_s_alive["pred_rel_l2"]))

    save_loss_curve(
        rollout_rel_l2, dirs["res"] / "rollout_rel_l2_vs_step.png",
        "AE+DMD Rollout Relative L2 Error vs Steps Beyond x1 (Full Grid)",
        xlabel="Steps beyond x1", ylabel="Relative L2 error", log_scale=False,
    )
    if rollout_rel_l2_alive:
        save_loss_curve(
            rollout_rel_l2_alive, dirs["res"] / "rollout_rel_l2_vs_step_alive.png",
            f"AE+DMD Rollout Relative L2 Error vs Steps Beyond x1 (Alive-Only, {int(np.count_nonzero(alive_grid))}/{alive_grid.size} px)",
            xlabel="Steps beyond x1", ylabel="Relative L2 error", log_scale=False,
        )


    ae_m = autoencoder_reconstruction_metrics(enc, dec, td.X, device)
    ae_m_alive = autoencoder_reconstruction_metrics_alive(enc, dec, td, device)
    dmd_m = dmd_one_step_metrics(enc, dec, dmd, td.X1, td.X2, device)
    pred_m = next_step_prediction_metrics(Z_pred, Z_true_next)
    print_metric_block("SINGLE MATRIX AE (RECON, FULL GRID)", ae_m)
    print_metric_block("SINGLE MATRIX AE (RECON, ALIVE-ONLY)", ae_m_alive)
    print_metric_block("SINGLE MATRIX DMD (ONE-STEP)", dmd_m)
    print_metric_block(f"SINGLE MATRIX PREDICT (+{k} FROM TRUE xT, FULL GRID)", pred_m)

    pred_m_alive = None
    if alive_grid is not None and bool(np.any(alive_grid)):
        pred_m_alive = next_step_prediction_metrics(Z_pred[alive_grid], Z_true_next[alive_grid])
        print_metric_block(f"SINGLE MATRIX PREDICT (+{k} FROM TRUE xT, ALIVE-ONLY)", pred_m_alive)


    metrics = {
        **ae_m, **ae_m_alive, **dmd_m, **pred_m,
        "predict_extra_steps": float(k), "dmd_spectral_radius": rho,
        **{f"rollout_final_{key}": val for key, val in rollout_final_m.items()},
        **rollout_step_metrics,
        "n_alive": float(td.meta.get("n_alive", -1)),
        "n_total": float(td.meta.get("n_total", -1)),
        **({f"rollout_final_alive_{key}": val for key, val in rollout_final_m_alive.items()}
           if rollout_final_m_alive is not None else {}),
        **({f"pred_alive_{key}": val for key, val in pred_m_alive.items()}
           if pred_m_alive is not None else {}),
    }
    write_metrics_txt(dirs["res"] / "metrics.txt", metrics)


if __name__ == "__main__":
    run_single_matrix()