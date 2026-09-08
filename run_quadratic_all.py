from __future__ import annotations
import os
import sys
import time
import runpy
import subprocess
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
QUAD_OUT = PROJECT / "quadratic_run"

MODES = [
    "ae-only-single",
    "ae-only-multi",
     "single-matrix",
     "multi-matrix",
     "dmd-only-single",
     "dmd-only-multi",
     "compare-architectures",
     "compare-architectures-multi",
]

QUADRATIC_RANK = None


def _run_child(mode: str) -> None:
    sys.path.insert(0, str(PROJECT))
    import defines as D
    D.AE_USE_QUADRATIC_PREDICTOR = True
    D.AE_QUADRATIC_RANK = QUADRATIC_RANK
    D.OUT_DIR = Path("out").resolve()
    sys.argv = ["main.py", mode]
    runpy.run_path(str(PROJECT / "main.py"), run_name="__main__")


def _orchestrate() -> None:
    QUAD_OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT) + os.pathsep + env.get("PYTHONPATH", "")

    results = []
    for mode in MODES:
        print(f"\n{'=' * 72}\n>>> {mode}   (quadratic predictor ON)\n{'=' * 72}", flush=True)
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child", mode],
            cwd=str(QUAD_OUT), env=env,
        )
        dt = time.time() - t0
        ok = proc.returncode == 0
        results.append((mode, ok, dt))
        status = "OK" if ok else f"FAILED (rc={proc.returncode})"
        print(f"<<< {mode}: {status}   ({dt:.0f}s)", flush=True)

    print(f"\n{'=' * 72}\nSUMMARY   (results under {QUAD_OUT / 'out'})\n{'=' * 72}")
    for mode, ok, dt in results:
        print(f"  {'ok  ' if ok else 'FAIL'}  {mode:30s}{dt:8.0f}s")
    if not all(ok for _, ok, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--child":
        _run_child(sys.argv[2])
    else:
        _orchestrate()