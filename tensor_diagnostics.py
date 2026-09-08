from __future__ import annotations

import torch


def check_tensor(t: torch.Tensor, name: str = "tensor") -> dict:
    info: dict = {"name": name, "shape": tuple(t.shape), "dtype": str(t.dtype)}

    finite_mask = torch.isfinite(t)
    is_finite = bool(finite_mask.all())
    info["finite"] = is_finite

    if not is_finite:
        has_nan = bool(torch.isnan(t).any())
        has_inf = bool(torch.isinf(t).any())
        n_bad = int((~finite_mask).sum())
        info.update(nan=has_nan, inf=has_inf, n_nonfinite=n_bad, numel=t.numel())

        print(f"[{name}] NON-FINITE  shape={info['shape']}  nan={has_nan}  "
              f"inf={has_inf}  bad_entries={n_bad}/{t.numel()}")

        if finite_mask.any():
            finite_vals = t[finite_mask]
            info["finite_min"] = float(finite_vals.min())
            info["finite_max"] = float(finite_vals.max())
            print(f"[{name}] remaining finite entries range: "
                  f"{info['finite_min']:.6g} to {info['finite_max']:.6g}")
        else:
            print(f"[{name}] every entry is non-finite.")

        return info

    t_min, t_max = float(t.min()), float(t.max())
    info["min"], info["max"] = t_min, t_max
    print(f"[{name}] finite  shape={info['shape']}  min={t_min:.6g}  max={t_max:.6g}")

    if t.ndim == 2:
        rank = int(torch.linalg.matrix_rank(t))
        smallest_sv = float(torch.linalg.svdvals(t)[-1])
        info["rank"] = rank
        info["smallest_singular_value"] = smallest_sv
        print(f"[{name}] rank={rank} of max-possible {min(t.shape)}   "
              f"smallest singular value={smallest_sv:.6g}")
    else:
        print(f"[{name}] ndim={t.ndim} != 2 -- skipping rank/singular value check")

    return info


if __name__ == "__main__":
    print("-- clean 2D tensor --")
    check_tensor(torch.randn(2048, 256), name="clean")

    print("\n-- tensor with a NaN and an Inf --")
    bad = torch.randn(8, 4)
    bad[0, 0] = float("nan")
    bad[1, 1] = float("inf")
    check_tensor(bad, name="broken")

    print("\n-- finite, no zeros, no infs, but nearly singular (rows nearly parallel) --")
    base = torch.randn(1, 4)
    near_singular = base.repeat(8, 1) + 1e-6 * torch.randn(8, 4)
    check_tensor(near_singular, name="near_singular")
