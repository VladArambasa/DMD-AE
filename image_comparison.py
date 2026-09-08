# ============================ image_comparison.py ===========================
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

import defines as D
from experiment_common import make_out_dirs
from data_loader import (
    load_all_A_matrices,
    split_explicit_matrix_indices,
)

try:
    _RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    _RESAMPLE = Image.LANCZOS


#CONFIGURATION
IMAGES_FRACTALS_DIR = D.PROJECT_DIR / "images fractals"

DEFAULT_REPORT_PATH = D.PROJECT_DIR / "image-comparison.txt"


#OBTAINED-IMAGE REGISTRY
_TRAIN_INDEXED_RE = re.compile(r"^train_gt_(final_mask|escape_iters)_(\d+)\.png$")

_PLAIN_KIND = {
    "gt_final_mask.png": "final_mask",
    "gt_escape_iters.png": "escape_iters",
}
_TRAIN_EXAMPLE_KIND = {
    "train_example_gt_final_mask.png": "final_mask",
    "train_example_gt_escape_iters.png": "escape_iters",
}
_TEST_KIND = {
    "test_gt_final_mask.png": "final_mask",
    "test_gt_escape_iters.png": "escape_iters",
}


@dataclass(frozen=True)
class ObtainedImage:
    path: Path
    mode: str
    index: int
    source: str
    kind: str
    role: str


def _resolve_multi_indices(source: str) -> tuple[np.ndarray, np.ndarray]:

    total = int(load_all_A_matrices(D.A_DATA_DIR, source=source).shape[0])
    return split_explicit_matrix_indices(
        total_count=total,
        train_count=D.MULTI_MATRIX_TRAIN_COUNT,
        test_count=D.MULTI_MATRIX_TEST_COUNT,
        seed=D.MULTI_MATRIX_SPLIT_SEED,
    )


def discover_obtained_images(mode: str) -> list[ObtainedImage]:

    root = Path("out") / mode
    if not root.exists():
        return []

    dirs = make_out_dirs(mode)
    is_multi = "multi" in mode
    source = D.MULTI_MATRIX_SOURCE if is_multi else D.SINGLE_MATRIX_SOURCE

    split_cache: tuple[np.ndarray, np.ndarray] | None = None
    found: list[ObtainedImage] = []

    for folder in (dirs["td"], dirs["res"]):
        if not folder.exists():
            continue
        for p in sorted(folder.glob("*.png")):
            name = p.name

            m = _TRAIN_INDEXED_RE.match(name)
            if m is not None:
                kind, idx_str = m.group(1), m.group(2)
                found.append(ObtainedImage(p, mode, int(idx_str), source, kind, "train"))
                continue

            if name in _PLAIN_KIND:
                found.append(ObtainedImage(p, mode, int(D.SINGLE_MATRIX_INDEX), source, _PLAIN_KIND[name], "single"))
                continue

            if name in _TRAIN_EXAMPLE_KIND:
                if split_cache is None:
                    split_cache = _resolve_multi_indices(source)
                train_idx, _test_idx = split_cache
                if train_idx.size:
                    found.append(ObtainedImage(p, mode, int(train_idx[0]), source, _TRAIN_EXAMPLE_KIND[name], "train-example"))
                continue

            if name in _TEST_KIND:
                if split_cache is None:
                    split_cache = _resolve_multi_indices(source)
                _train_idx, test_idx = split_cache
                if test_idx.size:
                    found.append(ObtainedImage(p, mode, int(test_idx[0]), source, _TEST_KIND[name], "test"))
                continue





    return found


def find_reference_image(images_root: Path, source: str, index: int) -> Path | None:

    task_dir = images_root / f"task-{source}"
    if not task_dir.exists():
        return None
    matches = sorted(task_dir.glob(f"exhibit-{index:02d}-*.png"))
    return matches[0] if matches else None


#IMAGE METRICS
def _otsu_threshold(gray: np.ndarray) -> float:

    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 127.0

    levels = np.arange(256, dtype=np.float64)
    sum_all = float(np.dot(levels, hist))

    weight_bg = 0.0
    sum_bg = 0.0
    best_thresh, best_var = 0, -1.0
    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg <= 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / weight_fg
        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > best_var:
            best_var = var_between
            best_thresh = t
    return float(best_thresh)


def _interior_mask(gray: np.ndarray, *, interior_is_bright: bool) -> np.ndarray:

    thresh = _otsu_threshold(gray)
    return (gray > thresh) if interior_is_bright else (gray <= thresh)


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a0 = a - a.mean()
    b0 = b - b.mean()
    denom = float(np.sqrt(np.sum(a0 * a0) * np.sum(b0 * b0)))
    if denom < 1e-9:
        return 0.0
    return float(np.sum(a0 * b0) / denom)


def _ssim(a: np.ndarray, b: np.ndarray, *, data_range: float = 255.0) -> float | None:

    try:
        from scipy.ndimage import uniform_filter
    except ImportError:
        return None

    a = a.astype(np.float64)
    b = b.astype(np.float64)
    k1, k2, win = 0.01, 0.03, 7
    c1, c2 = (k1 * data_range) ** 2, (k2 * data_range) ** 2

    mu_a, mu_b = uniform_filter(a, win), uniform_filter(b, win)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    var_a = uniform_filter(a * a, win) - mu_a2
    var_b = uniform_filter(b * b, win) - mu_b2
    cov_ab = uniform_filter(a * b, win) - mu_ab

    num = (2 * mu_ab + c1) * (2 * cov_ab + c2)
    den = (mu_a2 + mu_b2 + c1) * (var_a + var_b + c2)
    return float(np.mean(num / den))


def _iou_dice_agreement(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[float, float, float]:
    inter = float(np.logical_and(mask_a, mask_b).sum())
    union = float(np.logical_or(mask_a, mask_b).sum())
    total_true = float(mask_a.sum() + mask_b.sum())
    iou = inter / union if union > 0 else 1.0
    dice = (2.0 * inter / total_true) if total_true > 0 else 1.0
    agreement = float(np.mean(mask_a == mask_b))
    return iou, dice, agreement


def _verdict(iou: float) -> str:
    if iou >= 0.75:
        return "looks alike"
    if iou >= 0.45:
        return "roughly alike"
    return "does NOT look alike"


#ONE COMPARISON
@dataclass
class ComparisonResult:
    obtained: ObtainedImage
    reference: Path
    obtained_size: tuple[int, int]
    reference_size: tuple[int, int]
    compare_size: tuple[int, int]
    mse: float
    ncc: float
    ssim: float | None
    iou: float
    dice: float
    mask_agreement: float
    obtained_alive_frac: float
    reference_alive_frac: float


def compare_one(obtained: ObtainedImage, reference: Path) -> ComparisonResult:
    img_o = Image.open(obtained.path).convert("L")
    img_r = Image.open(reference).convert("L")
    size_o, size_r = img_o.size, img_r.size

    target = (min(size_o[0], size_r[0]), min(size_o[1], size_r[1]))
    if img_o.size != target:
        img_o = img_o.resize(target, _RESAMPLE)
    if img_r.size != target:
        img_r = img_r.resize(target, _RESAMPLE)

    a = np.asarray(img_o, dtype=np.float64)
    b = np.asarray(img_r, dtype=np.float64)

    interior_is_bright = obtained.kind == "final_mask"
    mask_a = _interior_mask(a, interior_is_bright=interior_is_bright)
    mask_b = _interior_mask(b, interior_is_bright=False)

    iou, dice, agreement = _iou_dice_agreement(mask_a, mask_b)

    return ComparisonResult(
        obtained=obtained,
        reference=reference,
        obtained_size=size_o,
        reference_size=size_r,
        compare_size=target,
        mse=_mse(a, b),
        ncc=_ncc(a, b),
        ssim=_ssim(a, b),
        iou=iou,
        dice=dice,
        mask_agreement=agreement,
        obtained_alive_frac=float(mask_a.mean()),
        reference_alive_frac=float(mask_b.mean()),
    )


#REPORT
def write_report(
    modes: list[str],
    results: list[ComparisonResult],
    missing: list[ObtainedImage],
    errors: list[str],
    images_root: Path,
    report_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("IMAGE COMPARISON REPORT")
    lines.append(f"generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"modes checked: {', '.join(modes) if modes else '(none)'}")
    lines.append(f"reference root: {images_root}")
    lines.append("")
    lines.append("metrics: MSE (0=identical, grayscale 0-255 scale) | NCC (normalized")
    lines.append("cross-correlation, -1..1, 1=identical up to brightness/contrast) | SSIM")
    lines.append("(structural similarity, 0..1, simplified/uniform-window) | interior-mask")
    lines.append("IoU/Dice (overlap of the bounded/'alive' blob after auto-thresholding both")
    lines.append("images, 0..1) -- IoU is the primary 'do these look alike' signal since it")
    lines.append("does not care about the colour/greyscale convention each image happens to")
    lines.append("use, only the shape of the bounded region.")
    lines.append("=" * 78)

    by_mode: dict[str, list[ComparisonResult]] = {}
    for r in results:
        by_mode.setdefault(r.obtained.mode, []).append(r)

    for mode in modes:
        mode_results = by_mode.get(mode, [])
        lines.append("")
        lines.append(f"[{mode}]")
        if not mode_results:
            lines.append("  no ground-truth PNGs with a matching reference were found")
            continue
        for r in sorted(mode_results, key=lambda r: (r.obtained.role, r.obtained.index, r.obtained.kind)):
            o = r.obtained
            lines.append("")
            lines.append(f"  {o.role} | task-{o.source} | matrix index {o.index:02d} | {o.kind}")
            lines.append(f"    obtained : {o.path}")
            lines.append(f"    reference: {r.reference}")
            lines.append(
                f"    sizes -- obtained={r.obtained_size[0]}x{r.obtained_size[1]}  "
                f"reference={r.reference_size[0]}x{r.reference_size[1]}  "
                f"compared_at={r.compare_size[0]}x{r.compare_size[1]}"
            )
            ssim_str = f"{r.ssim:.4f}" if r.ssim is not None else "n/a (scipy not found)"
            lines.append(f"    MSE={r.mse:.4f}  NCC={r.ncc:+.4f}  SSIM={ssim_str}")
            lines.append(
                f"    interior-mask IoU={r.iou:.4f}  Dice={r.dice:.4f}  "
                f"pixel-agreement={r.mask_agreement:.4f}  -->  {_verdict(r.iou)}"
            )
            lines.append(
                f"    bounded-area fraction -- obtained={r.obtained_alive_frac:.4f}  "
                f"reference={r.reference_alive_frac:.4f}"
            )

    if missing:
        lines.append("")
        lines.append("=" * 78)
        lines.append("NO REFERENCE IMAGE FOUND FOR:")
        for om in missing:
            lines.append(
                f"  [{om.mode}/{om.role}] {om.path}  "
                f"(expected task-{om.source} exhibit-{om.index:02d}-*.png under {images_root})"
            )

    if errors:
        lines.append("")
        lines.append("=" * 78)
        lines.append("ERRORS WHILE COMPARING:")
        for e in errors:
            lines.append(f"  {e}")

    if not results and not missing and not errors:
        lines.append("")
        lines.append("Nothing to compare -- run main.py for at least one mode first.")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


#ORCHESTRATION
def _collect(modes: list[str], images_root: Path) -> tuple[list[ComparisonResult], list[ObtainedImage], list[str]]:

    results: list[ComparisonResult] = []
    missing: list[ObtainedImage] = []
    errors: list[str] = []

    for mode in modes:
        for obtained in discover_obtained_images(mode):
            reference = find_reference_image(images_root, obtained.source, obtained.index)
            if reference is None:
                missing.append(obtained)
                continue
            try:
                results.append(compare_one(obtained, reference))
            except Exception as exc:
                errors.append(f"{obtained.path}: {exc!r}")

    return results, missing, errors


def _summarize(results: list[ComparisonResult], missing: list[ObtainedImage], errors: list[str]) -> str:
    if not results and not missing and not errors:
        return "no ground-truth images found to check (did main() actually save any yet?)"
    parts = []
    if results:
        n_alike = sum(1 for r in results if _verdict(r.iou) == "looks alike")
        parts.append(f"{n_alike}/{len(results)} look alike (interior-mask IoU)")
    if missing:
        parts.append(f"{len(missing)} missing a reference exhibit")
    if errors:
        parts.append(f"{len(errors)} failed to compare")
    return ", ".join(parts)


def run_image_comparison(
    modes: list[str],
    *,
    images_root: Path = IMAGES_FRACTALS_DIR,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> Path:
    results, missing, errors = _collect(modes, images_root)
    write_report(modes, results, missing, errors, images_root, report_path)
    return report_path


def check_experiment_images(
    experiment: str,
    *,
    images_root: Path = IMAGES_FRACTALS_DIR,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> Path:

    results, missing, errors = _collect([experiment], images_root)
    write_report([experiment], results, missing, errors, images_root, report_path)
    print(f"[image_comparison] {experiment}: {_summarize(results, missing, errors)} -- see {report_path}")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the ground-truth fractal PNGs under out/<mode>/ "
                     "against the reference exhibit images in the images-fractals "
                     "folder, matched by task and matrix index.",
    )
    parser.add_argument(
        "mode", nargs="?", default=None, choices=list(D.COMMANDS),
        help="which experiment's output to check (default: every mode that currently has output under out/)",
    )
    parser.add_argument(
        "--images-root", type=Path, default=IMAGES_FRACTALS_DIR,
        help=f"folder containing task-emotion/ and task-rest/ exhibit images (default: {IMAGES_FRACTALS_DIR})",
    )
    parser.add_argument(
        "--report", type=Path, default=DEFAULT_REPORT_PATH,
        help=f"where to write the report (default: {DEFAULT_REPORT_PATH})",
    )
    args = parser.parse_args()

    modes = [args.mode] if args.mode is not None else [m for m in D.COMMANDS if (Path("out") / m).exists()]
    if not modes:
        print("Nothing found under out/ yet -- run main.py for at least one experiment first.")
        return

    report_path = run_image_comparison(modes, images_root=args.images_root, report_path=args.report)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()