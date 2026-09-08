from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_tensor(x: Any, device: torch.device) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


def save_model(model: torch.nn.Module, path: str | os.PathLike[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(path))


def load_model(model: torch.nn.Module, path: str | os.PathLike[str], device: torch.device):
    model.load_state_dict(torch.load(path, map_location=device))
    return model



def set_recommended_matplotlib() -> None:
    try:
        import matplotlib.pyplot as mp
    except Exception:
        return

    defaults: dict[str, dict[str, Any]] = {
        "figure": {"figsize": (16, 8), "dpi": 300, "constrained_layout.use": True},
        "legend": {"fontsize": 20},
        "lines": {"linewidth": 2, "markersize": 5},
        "axes": {
            "labelsize": 28,
            "titlesize": 28,
            "grid": True,
            "grid.axis": "both",
            "grid.which": "both",
            "prop_cycle": mp.rcParams.get("axes.prop_cycle", None),
        },
        "xtick": {"labelsize": 20, "direction": "inout"},
        "ytick": {"labelsize": 20, "direction": "inout"},
        "xtick.major": {"size": 6.5, "width": 1.5},
        "ytick.major": {"size": 6.5, "width": 1.5},
        "xtick.minor": {"size": 4.0},
        "ytick.minor": {"size": 4.0},
    }

    try:
        import scienceplots
        mp.style.use(["science", "ieee"])
    except Exception:
        pass

    for group, params in defaults.items():
        if group == "axes" and params.get("prop_cycle", None) is None:
            params = {k: v for k, v in params.items() if k != "prop_cycle"}
        mp.rc(group, **params)


@contextmanager
def axis(filename: str | os.PathLike[str]) -> Iterator[Any]:
    import matplotlib.pyplot as mp

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    if not filename.suffix:
        ext = mp.rcParams.get("savefig.format", "png")
        filename = filename.with_suffix(f".{ext}")

    fig = mp.figure()
    ax = fig.gca()

    try:
        yield ax
    finally:
        fig.savefig(str(filename))
        mp.close(fig)

from pathlib import Path
import numpy as np

def _to_u8(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img)
    if a.dtype == np.uint8:
        return a
    if a.dtype == bool:
        return (a.astype(np.uint8) * 255)

    a = a.astype(np.float32, copy=False)
    amin = float(np.min(a))
    amax = float(np.max(a))
    if amax - amin < 1e-12:
        return np.zeros(a.shape, dtype=np.uint8)

    a = (a - amin) / (amax - amin)
    return (a * 255.0).clip(0, 255).astype(np.uint8)

def save_gray_png(img: np.ndarray, path) -> str:
    from PIL import Image

    path = str(path)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    u8 = _to_u8(img)
    if u8.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {u8.shape}")

    Image.fromarray(u8, mode="L").save(path)
    return path

def show_image(path: str) -> None:
    from PIL import Image
    Image.open(path).show()

