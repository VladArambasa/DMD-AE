from __future__ import annotations

import argparse
import sys
from datetime import datetime

import defines as D
from experiment_common import pick_device
from run_single_matrix import run_single_matrix
from run_multi_matrix import run_multi_matrix
from extra_optional import (
    run_ae_only_single,
    run_ae_only_multi,
    run_dmd_only_single,
    run_dmd_only_multi,
)
from image_comparison import check_experiment_images
from compare_architectures import run_comparison
from compare_architectures_multi import run_comparison_multi

class _Tee:
    def __init__(self, stream, logfile):
        self._stream = stream
        self._logfile = logfile

    def write(self, data):
        self._stream.write(data)
        self._logfile.write(data)
        self._logfile.flush()
        return len(data)

    def flush(self):
        self._stream.flush()
        self._logfile.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _start_console_log(parameter: str):
    console_dir = D.OUT_DIR / "console"
    console_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d-%H")
    logpath = console_dir / f"{parameter}-{stamp}.txt"

    logfile = open(logpath, "a", encoding="utf-8")
    logfile.write(f"\n===== {parameter} @ {datetime.now():%Y-%m-%d %H:%M:%S} =====\n")

    saved_out, saved_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(saved_out, logfile)
    sys.stderr = _Tee(saved_err, logfile)

    def restore():
        sys.stdout, sys.stderr = saved_out, saved_err
        logfile.flush()
        logfile.close()

    print(f"[console log] saving this run to {logpath}")
    return restore


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one dissertation experiment")
    parser.add_argument(
        "experiment",
        choices=[
            "single-matrix",
            "multi-matrix",
            "ae-only-single",
            "ae-only-multi",
            "dmd-only-single",
            "dmd-only-multi",
            "compare-architectures",  # Lusch / DLKoopman / DLDMD, single matrix
            "compare-architectures-multi",  # Lusch / DLKoopman / DLDMD, multi-matrix
        ],
        help="choose exactly one experiment to run",
    )
    parser.add_argument(
        "--epochs", type=int, default=60,
        help="epochs for compare-architectures (ignored by the other modes)",
    )
    args = parser.parse_args()

    restore = _start_console_log(args.experiment)
    try:
        device = pick_device()
        print("DEVICE:", device)

        if args.experiment == "compare-architectures":
            run_comparison(epochs=args.epochs)
            return
        elif args.experiment == "compare-architectures-multi":
            run_comparison_multi(epochs=args.epochs)
            return
        elif args.experiment == "single-matrix":
            run_single_matrix(device)
        elif args.experiment == "multi-matrix":
            run_multi_matrix(device)
        elif args.experiment == "ae-only-single":
            run_ae_only_single(device)
        elif args.experiment == "ae-only-multi":
            run_ae_only_multi(device)
        elif args.experiment == "dmd-only-single":
            run_dmd_only_single(device)
        elif args.experiment == "dmd-only-multi":
            run_dmd_only_multi(device)

        check_experiment_images(args.experiment)
    finally:
        restore()


if __name__ == "__main__":
    main()