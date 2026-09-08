from __future__ import annotations

import os
import numpy as np
import torch

import defines as D
from utils import save_model
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
    make_out_dirs,
    fit_streamed_dmd_from_td_list,
    print_metric_block,
    mean_metric_dict,
    write_metrics_txt,
)


def run_multi_matrix(device: torch.device | None = None) -> None:
    if device is None:
        device = pick_device()

    print("DEVICE:", device)
    print("CWD:", os.getcwd())

    dirs = make_out_dirs("multi-matrix")

    print("\n================ MULTI MATRIX TRAIN / TEST ================\n")

    A_all = load_all_A_matrices(D.A_DATA_DIR, source=D.MULTI_MATRIX_SOURCE)
    total_matrices = int(A_all.shape[0])
    print("TOTAL MATRICES:", total_matrices)

    train_idx, test_idx = split_explicit_matrix_indices(
        total_count=total_matrices,
        train_count=D.MULTI_MATRIX_TRAIN_COUNT,
        test_count=D.MULTI_MATRIX_TEST_COUNT,
        seed=D.MULTI_MATRIX_SPLIT_SEED,
    )
    print("TRAIN COUNT:", int(train_idx.size), "TEST COUNT:", int(test_idx.size))
    print("TRAIN IDX:", train_idx.tolist())
    print("TEST IDX :", test_idx.tolist())


    td_train_list = build_matrix_c_grid_training_data_many_matrices(
        data_dir=D.A_DATA_DIR,
        source=D.MULTI_MATRIX_SOURCE,
        indices=train_idx,
        c_re_min=D.C_RE_MIN,
        c_re_max=D.C_RE_MAX,
        c_im_min=D.C_IM_MIN,
        c_im_max=D.C_IM_MAX,
        c_re_n=D.MULTI_MATRIX_C_RE_N,
        c_im_n=D.MULTI_MATRIX_C_IM_N,
        max_iters=D.TRAIN_MAX_ITERS,
        escape_r=D.DYNAMICS_CLAMP_R,
        classify_r=D.ESCAPE_R,
        filter_escaped=D.FILTER_ESCAPED_FOR_TRAINING,
    )

    for i, td in enumerate(td_train_list):
        matrix_id = int(train_idx[i])
        save_ground_truth_final_mask(
            td, D.ESCAPE_R,
            dirs["td"] / f"train_gt_final_mask_{matrix_id:02d}.png",
            scale=D.IMAGE_SCALE,
        )
        save_ground_truth_escape_iters(
            td, D.ESCAPE_R,
            dirs["td"] / f"train_gt_escape_iters_{matrix_id:02d}.png",
        )

    for td in td_train_list:
        td.X_grid = None


    enc, dec, losses, loss_components, val_losses = train_autoencoder(
        [td.X1 for td in td_train_list],
        [td.X2 for td in td_train_list],
        latent_dim=D.LATENT_DIM,
        epochs=D.AE_EPOCHS,
        batch_size=D.AE_BATCH_SIZE,
        lr=D.AE_LR,
        device=device,
    )

    has_val = bool(np.any(np.isfinite(val_losses))) if len(val_losses) else False
    save_loss_curve(
        losses, dirs["res"] / "loss_curve.png", "Multi-Matrix AE Loss",
        extra_series=({"Validation": val_losses} if has_val else None),
        primary_label="Train" if has_val else None,
    )
    save_loss_curve(loss_components["rec"], dirs["res"] / "loss_curve_rec.png",
                    "Multi-Matrix AE Loss -- Reconstruction Component")
    save_loss_curve(loss_components["lin"], dirs["res"] / "loss_curve_lin.png",
                    "Multi-Matrix AE Loss -- Latent Linearity Component")
    save_loss_curve(loss_components["pred"], dirs["res"] / "loss_curve_pred.png",
                    "Multi-Matrix AE Loss -- Decoded Prediction Component")
    save_model(enc, os.path.join(D.CHECKPOINT_DIR, "encoder_multi_matrix.pth"))
    save_model(dec, os.path.join(D.CHECKPOINT_DIR, "decoder_multi_matrix.pth"))

    dmd = fit_streamed_dmd_from_td_list(enc, td_train_list, device)
    rho = float(np.max(np.abs(np.linalg.eigvals(dmd.A.detach().cpu().numpy()))))
    print("MULTI DMD SPECTRAL RADIUS:", rho)

    ae_train_metrics_all, dmd_train_metrics_all = [], []
    for i, td in enumerate(td_train_list):
        ae_tr = autoencoder_reconstruction_metrics_alive(enc, dec, td, device)
        dmd_tr = dmd_one_step_metrics(enc, dec, dmd, td.X1, td.X2, device)
        ae_train_metrics_all.append(ae_tr)
        dmd_train_metrics_all.append(dmd_tr)
        print_metric_block(f"TRAIN MATRIX {int(train_idx[i])} AE (RECON, ALIVE-ONLY)", ae_tr)
        print_metric_block(f"TRAIN MATRIX {int(train_idx[i])} DMD (ONE-STEP)", dmd_tr)

    mean_ae_train = mean_metric_dict(ae_train_metrics_all)
    mean_dmd_train = mean_metric_dict(dmd_train_metrics_all)
    print_metric_block("MEAN TRAIN AE (RECON, ALIVE-ONLY)", mean_ae_train)
    print_metric_block("MEAN TRAIN DMD (ONE-STEP)", mean_dmd_train)
    write_metrics_txt(
        dirs["res"] / "mean_train_metrics.txt",
        {
            **{f"train_{k2}": v for k2, v in mean_ae_train.items()},
            **{f"train_{k2}": v for k2, v in mean_dmd_train.items()},
            "n_train_matrices": float(len(td_train_list)),
        },
    )


    td_test_list = build_matrix_c_grid_training_data_many_matrices(
        data_dir=D.A_DATA_DIR,
        source=D.MULTI_MATRIX_SOURCE,
        indices=test_idx,
        c_re_min=D.C_RE_MIN,
        c_re_max=D.C_RE_MAX,
        c_im_min=D.C_IM_MIN,
        c_im_max=D.C_IM_MAX,
        c_re_n=D.MULTI_MATRIX_C_RE_N,
        c_im_n=D.MULTI_MATRIX_C_IM_N,
        max_iters=D.TRAIN_MAX_ITERS,
        escape_r=D.DYNAMICS_CLAMP_R,
        classify_r=D.ESCAPE_R,
        filter_escaped=D.FILTER_ESCAPED_FOR_TRAINING,
    )

    k = int(D.PREDICT_EXTRA_STEPS)
    ae_metrics_all = []
    ae_alive_metrics_all = []
    dmd_metrics_all = []
    pred_metrics_all = []
    pred_alive_metrics_all = []

    maxit = int(D.TRAIN_MAX_ITERS)
    n_check = min(int(getattr(D, "PREDICT_ROLLOUT_CHECK_STEPS", 10)), max(maxit - 1, 0))
    rollout_final_metrics_all = []
    rollout_final_alive_metrics_all = []
    rollout_step_curves_all = []
    rollout_step_curves_alive_all = []

    for j, td_test in enumerate(td_test_list):
        A_test = A_all[int(test_idx[j])]
        alive_grid = td_test.meta.get("alive_mask_grid", None)

        ae_m = autoencoder_reconstruction_metrics(enc, dec, td_test.X, device)
        ae_m_alive = autoencoder_reconstruction_metrics_alive(enc, dec, td_test, device)
        dmd_m = dmd_one_step_metrics(enc, dec, dmd, td_test.X1, td_test.X2, device)


        Z_pred = predict_next_snapshot(td_test, enc, dec, dmd, device, steps=k, escape_r=D.DYNAMICS_CLAMP_R)
        Z_true_next = iterate_true_next_snapshot(td_test, A_test, steps=k, escape_r=D.DYNAMICS_CLAMP_R)
        pred_m = next_step_prediction_metrics(Z_pred, Z_true_next)

        pred_m_alive = None
        if alive_grid is not None and bool(np.any(alive_grid)):
            pred_m_alive = next_step_prediction_metrics(Z_pred[alive_grid], Z_true_next[alive_grid])


        rollout = predict_rollout_from_start_ae_dmd(
            td_test, enc, dec, dmd, device, steps=maxit, escape_r=D.DYNAMICS_CLAMP_R,
        )
        d_state = int((int(td_test.X_grid.shape[-1]) - 2) // 2)

        Z_pred_final = rollout[-1]
        Z_true_final = td_test.X_grid[-1][..., :2 * d_state]
        final_m = next_step_prediction_metrics(Z_pred_final, Z_true_final)
        rollout_final_metrics_all.append(final_m)

        if alive_grid is not None and bool(np.any(alive_grid)):
            final_m_alive = next_step_prediction_metrics(Z_pred_final[alive_grid], Z_true_final[alive_grid])
            rollout_final_alive_metrics_all.append(final_m_alive)
            print_metric_block(
                f"TEST MATRIX {int(test_idx[j])} AE+DMD ROLLOUT x1 -> x{maxit} (MACRO, ALIVE-ONLY, "
                f"{int(np.count_nonzero(alive_grid))}/{alive_grid.size} px)", final_m_alive,
            )

        step_curve = []
        step_curve_alive = []
        for s in range(n_check):
            Z_pred_s = rollout[s]
            Z_true_s = td_test.X_grid[s + 1][..., :2 * d_state]
            m_s = next_step_prediction_metrics(Z_pred_s, Z_true_s)
            step_curve.append(float(m_s["pred_rel_l2"]))

            if alive_grid is not None and bool(np.any(alive_grid)):
                m_s_alive = next_step_prediction_metrics(Z_pred_s[alive_grid], Z_true_s[alive_grid])
                step_curve_alive.append(float(m_s_alive["pred_rel_l2"]))

        rollout_step_curves_all.append(step_curve)
        if step_curve_alive:
            rollout_step_curves_alive_all.append(step_curve_alive)


        ae_metrics_all.append(ae_m)
        ae_alive_metrics_all.append(ae_m_alive)
        dmd_metrics_all.append(dmd_m)
        pred_metrics_all.append(pred_m)
        if pred_m_alive is not None:
            pred_alive_metrics_all.append(pred_m_alive)

        print_metric_block(f"TEST MATRIX {int(test_idx[j])} AE (RECON, FULL GRID)", ae_m)
        print_metric_block(f"TEST MATRIX {int(test_idx[j])} AE (RECON, ALIVE-ONLY)", ae_m_alive)
        print_metric_block(f"TEST MATRIX {int(test_idx[j])} DMD (ONE-STEP)", dmd_m)
        print_metric_block(f"TEST MATRIX {int(test_idx[j])} PREDICT (+{k}, FULL GRID)", pred_m)
        if pred_m_alive is not None:
            print_metric_block(f"TEST MATRIX {int(test_idx[j])} PREDICT (+{k}, ALIVE-ONLY)", pred_m_alive)
        print_metric_block(f"TEST MATRIX {int(test_idx[j])} AE+DMD ROLLOUT x1 -> x{maxit} (MACRO, FULL GRID)", final_m)

        if j == 0:
            save_ground_truth_final_mask(td_test, D.ESCAPE_R, dirs["res"] / "test_gt_final_mask.png", scale=D.IMAGE_SCALE)
            save_ground_truth_escape_iters(td_test, D.ESCAPE_R, dirs["res"] / "test_gt_escape_iters.png")

            Z_recon = reconstruct_true_final_snapshot(td_test, enc, dec, device)
            save_final_snapshot_image(Z_recon, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "test_recon_final_mask.png",
                                       mode="mask", alive_mask=alive_grid)
            save_final_snapshot_image(Z_recon, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "test_recon_final_snapshot_mag.png",
                                       mode="mag", alive_mask=alive_grid)

            save_final_snapshot_image(Z_pred, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "test_pred_final_mask.png",
                                       mode="mask", alive_mask=alive_grid)
            save_final_snapshot_image(Z_pred, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "test_pred_final_snapshot_mag.png",
                                       mode="mag", alive_mask=alive_grid)
            save_final_snapshot_image(Z_true_next, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "test_true_next_final_mask.png", mode="mask")
            save_final_snapshot_image(Z_true_next, escape_r=D.ESCAPE_R, out_png=dirs["res"] / "test_true_next_final_snapshot_mag.png", mode="mag")

            iters_pred = teacher_forced_escape_iters(td_test, enc, dec, dmd, device, escape_r=D.ESCAPE_R)
            save_escape_image(iters_pred, max_iters=int(td_test.X_grid.shape[0]), out_png=dirs["res"] / "test_pred_escape_iters.png",
                               alive_mask=alive_grid)


            save_final_snapshot_image(Z_pred_final, escape_r=D.ESCAPE_R,
                                       out_png=dirs["res"] / "test_rollout_from_start_final_mask.png",
                                       mode="mask", alive_mask=alive_grid)
            save_final_snapshot_image(Z_pred_final, escape_r=D.ESCAPE_R,
                                       out_png=dirs["res"] / "test_rollout_from_start_final_mag.png",
                                       mode="mag", alive_mask=alive_grid)

    mean_ae = mean_metric_dict(ae_metrics_all)
    mean_ae_alive = mean_metric_dict(ae_alive_metrics_all)
    mean_dmd = mean_metric_dict(dmd_metrics_all)
    mean_pred = mean_metric_dict(pred_metrics_all)
    print_metric_block("MEAN TEST AE (RECON, FULL GRID)", mean_ae)
    print_metric_block("MEAN TEST AE (RECON, ALIVE-ONLY)", mean_ae_alive)
    print_metric_block("MEAN TEST DMD (ONE-STEP)", mean_dmd)
    print_metric_block(f"MEAN TEST PREDICT (+{k}, FULL GRID)", mean_pred)

    mean_pred_alive = {}
    if pred_alive_metrics_all:
        mean_pred_alive_raw = mean_metric_dict(pred_alive_metrics_all)
        print_metric_block(f"MEAN TEST PREDICT (+{k}, ALIVE-ONLY)", mean_pred_alive_raw)
        mean_pred_alive = {f"pred_alive_{key}": val for key, val in mean_pred_alive_raw.items()}


    mean_rollout_final = mean_metric_dict(rollout_final_metrics_all)
    print_metric_block(f"MEAN TEST AE+DMD ROLLOUT x1 -> x{maxit} (MACRO, FULL GRID)", mean_rollout_final)
    mean_rollout_final_alive = {}
    if rollout_final_alive_metrics_all:
        mean_rollout_final_alive_raw = mean_metric_dict(rollout_final_alive_metrics_all)
        print_metric_block(f"MEAN TEST AE+DMD ROLLOUT x1 -> x{maxit} (MACRO, ALIVE-ONLY)", mean_rollout_final_alive_raw)
        mean_rollout_final_alive = {f"rollout_final_alive_{key}": val for key, val in mean_rollout_final_alive_raw.items()}

    rollout_step_metrics: dict[str, float] = {}
    mean_step_curve: list[float] = []
    if rollout_step_curves_all:
        mean_step_curve = np.mean(np.asarray(rollout_step_curves_all, dtype=np.float64), axis=0).tolist()
        for s, val in enumerate(mean_step_curve):
            rollout_step_metrics[f"rollout_rel_l2_step_{s + 1:03d}"] = float(val)

    mean_step_curve_alive: list[float] = []
    if rollout_step_curves_alive_all:
        mean_step_curve_alive = np.mean(np.asarray(rollout_step_curves_alive_all, dtype=np.float64), axis=0).tolist()
        for s, val in enumerate(mean_step_curve_alive):
            rollout_step_metrics[f"rollout_rel_l2_alive_step_{s + 1:03d}"] = float(val)

    if mean_step_curve:
        save_loss_curve(
            mean_step_curve, dirs["res"] / "rollout_rel_l2_vs_step.png",
            f"AE+DMD Rollout Relative L2 Error vs Steps Beyond x1 (Mean Over {len(rollout_step_curves_all)} Test Matrices, Full Grid)",
            xlabel="Steps beyond x1", ylabel="Relative L2 error", log_scale=False,
        )
    if mean_step_curve_alive:
        save_loss_curve(
            mean_step_curve_alive, dirs["res"] / "rollout_rel_l2_vs_step_alive.png",
            f"AE+DMD Rollout Relative L2 Error vs Steps Beyond x1 (Mean Over {len(rollout_step_curves_alive_all)} Test Matrices, Alive-Only)",
            xlabel="Steps beyond x1", ylabel="Relative L2 error", log_scale=False,
        )


    write_metrics_txt(
        dirs["res"] / "mean_test_metrics.txt",
        {
            **mean_ae, **mean_ae_alive, **mean_dmd, **mean_pred, **mean_pred_alive,
            "predict_extra_steps": float(k), "dmd_spectral_radius": rho,
            **{f"rollout_final_{key}": val for key, val in mean_rollout_final.items()},
            **mean_rollout_final_alive,
            **rollout_step_metrics,
        },
    )


if __name__ == "__main__":
    run_multi_matrix()