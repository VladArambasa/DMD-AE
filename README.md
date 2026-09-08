# Nonlinear Dynamics Approximation Using Dynamic Mode Decomposition and Autoencoders

Software for the MSc thesis titled: #Nonlinear Dynamics Approximation Using Dynamic Mode Decomposition and Autoencoders  (Vlad Arambașa, West University of Timișoara, 2026,
supervisors prof. univ. dr. Eva Kaslik and C.S. III dr. Alexandru Fikl). A hybrid autoencoder + DMD
pipeline for the matrix-coupled Mandelbrot map `z_{n+1} = (A z_n)² + c`, applied to brain connectomes.
See the thesis for the full background, methodology and results.

## Requirements

- Python 3.10+ (developed on 3.12)
- Packages: `torch`, `numpy`, `scipy`, `pillow`, `matplotlib`

## Install

```bash
git clone https://github.com/UVT-ARAMBASA/UVT-DISERTATIE-7E9.git
cd UVT-DISERTATIE-7E9

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install torch numpy scipy pillow matplotlib
# or, if the repo has one:  python -m pip install -r requirements.txt
```

## Data

Place the two connectome archives in `data/` before running (not included — obtain from the project
coordinators):

```
data/task-emotion.npz
data/task-rest.npz
```

## Run the experiments

Each experiment is a single command — data generation, training and evaluation all happen inside it.
Run exactly one mode:

```bash
python main.py single-matrix                 # full AE+DMD pipeline, one connectome
python main.py multi-matrix                  # shared model on a 10/2 held-out split (overnight on CPU)
python main.py ae-only-single                # AE-only baseline
python main.py ae-only-multi
python main.py dmd-only-single               # DMD-only baseline
python main.py dmd-only-multi
python main.py compare-architectures         # Lusch / DLKoopman / DLDMD, single matrix
python main.py compare-architectures-multi   # the same three, held-out split
```

The only command-line flag is `--epochs` (used by the `compare-architectures` modes, default 60):

```bash
python main.py compare-architectures --epochs 60
```

Everything else — data source (`emotion`/`rest`), matrix index, latent dimension, `c`-grid, rollout
horizon, DMD ridge, seeds — is set in `defines.py`. To run any mode with the quadratic predictor
instead of the generic autoencoder, set `AE_USE_QUADRATIC_PREDICTOR = True` in `defines.py` (or run
`python run_quadratic_all.py` to sweep every mode).

## Outputs

Results are written to `out/<mode>/`:

- `results/` — `metrics.txt` plus mask, magnitude, and loss-curve figures

## Data set:

Coordinator's data-set is available at:
https://github.com/alexfikl/2025-fractal-connectomes-paper-experiments
- `training-data/` — the cached `(X₁, X₂)` arrays and ground-truth masks

A GPU is used automatically when available (`USE_CUDA_IF_AVAILABLE` in `defines.py`); otherwise the CPU
is used. Re-running a mode overwrites its `out/<mode>/` folder, so archive anything worth keeping first.
