"""Small I/O and bookkeeping helpers shared by the experiments."""

from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def out_dir(experiment: str, preset: str, base: str | Path | None = None) -> Path:
    d = Path(base) if base else ROOT / "results" / experiment / preset
    d.mkdir(parents=True, exist_ok=True)
    return d


def environment() -> dict:
    """Software / hardware provenance stored next to every result."""
    env = {"python": platform.python_version(), "platform": platform.platform(),
           "numpy": np.__version__, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except Exception:                                    # torch is optional for the exact theory
        pass
    try:
        env["git"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                             stderr=subprocess.DEVNULL).decode().strip()
        env["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
            stderr=subprocess.DEVNULL).decode().strip())
    except Exception:
        env["git"] = None
    return env


class _Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return o.item()
        try:
            import torch
            if isinstance(o, torch.Tensor):
                return o.detach().cpu().tolist()
        except Exception:
            pass
        return super().default(o)


def save_json(path: Path, obj: dict) -> None:
    Path(path).write_text(json.dumps(obj, indent=1, cls=_Encoder))


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def set_style() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 200, "font.size": 10, "axes.grid": True,
        "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "lines.linewidth": 1.6, "figure.constrained_layout.use": True,
    })


# colour-blind-safe palette (Okabe-Ito)
COLORS = {"mle": "#000000", "tree": "#D55E00", "oneloop": "#0072B2", "profiled": "#CC79A7",
          "learned": "#009E73", "pinn": "#E69F00", "adam": "#56B4E9", "grey": "#999999"}
