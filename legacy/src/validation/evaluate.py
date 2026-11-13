"""Validate trained PINN models against the analytic heat-equation solution.

This script requires no training: it loads the saved .keras models and
measures (1) L2 relative error against the exact solution, (2) the physics
residuals reported by the loss functions, and (3) pointwise error maps.

Usage (from src/):  python -m validation.evaluate
"""

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from validation.analytic import exact_solution, ic_field, gaussian_factor
from data_generation.generate_1d import generate_1d_data, gaussian_initial_condition
from data_generation.generate_2d import generate_2d_square, gaussian_initial_condition_2d
from data_generation.generate_3d import generate_3d_cube, gaussian_initial_condition_3d
from training.losses import (
    heat_1d_loss, boundary_loss, heat_2d_loss, boundary_loss_2d,
    heat_3d_loss, boundary_loss_3d, initial_condition_loss,
)

TIMES = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
OUT_DIR = ROOT / "results" / "validation"

CONFIGS = {
    1: dict(L=1.0, T=1.0, alpha=0.1, amplitude=1.0, width=0.1, grid=201),
    2: dict(L=1.0, T=1.0, alpha=0.1, amplitude=1.5, width=0.25, grid=101),
    3: dict(L=1.0, T=1.0, alpha=0.1, amplitude=1.5, width=0.25, grid=61),
}


def _model_predictor(dim, model, grid):
    """Return predict(t) evaluating the model on a uniform grid at time t."""
    xs = np.linspace(-CONFIGS[dim]["L"], CONFIGS[dim]["L"], grid)

    if dim == 1:
        def predict(t):
            pts = np.stack([xs, np.full(grid, t)], axis=1)
            return model.predict(pts, verbose=0)[:, 0]

    elif dim == 2:
        X, Y = np.meshgrid(xs, xs)

        def predict(t):
            pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, t)], axis=1)
            return model.predict(pts, verbose=0)[:, 0].reshape(grid, grid)

    else:
        X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")

        def predict(t):
            pts = np.stack([X.ravel(), Y.ravel(), Z.ravel(), np.full(X.size, t)], axis=1)
            return model.predict(pts, verbose=0)[:, 0].reshape(grid, grid, grid)

    return xs, predict


def _residuals(dim, model, cfg):
    """Physics residuals (PDE, boundary, initial condition) of the model."""
    L, T, alpha = cfg["L"], cfg["T"], cfg["alpha"]
    if dim == 1:
        data = generate_1d_data(9000, L, T)
        pde = heat_1d_loss(model, data["collocation"], alpha)
        bc = boundary_loss(model, data["boundary"], L)
        ic_target = gaussian_initial_condition(data["initial"][:, 0:1])
    elif dim == 2:
        data = generate_2d_square(9000, L, T)
        pde = heat_2d_loss(model, data["collocation"], alpha)
        bc = boundary_loss_2d(model, data["boundary"], L)
        ic_target = gaussian_initial_condition_2d(
            data["initial"][:, 0:1], data["initial"][:, 1:2])
    else:
        data = generate_3d_cube(9000, L, T)
        pde = heat_3d_loss(model, data["collocation"], alpha)
        bc = boundary_loss_3d(model, data["boundary"], L)
        ic_target = gaussian_initial_condition_3d(
            data["initial"][:, 0:1], data["initial"][:, 1:2],
            data["initial"][:, 2:3])

    ic = initial_condition_loss(model, data["initial"], ic_target)
    return {
        "pde_rms": float(tf.sqrt(pde).numpy()),
        "bc_rms": float(tf.sqrt(bc).numpy()),
        "ic_rms": float(tf.sqrt(ic).numpy()),
    }


def evaluate_dim(dim):
    """Return (rows, series_check, residuals, predict) for one dimension."""
    cfg = CONFIGS[dim]
    L, alpha = cfg["L"], cfg["alpha"]
    model = tf.keras.models.load_model(ROOT / "saved_models" / f"pinn_{dim}d.keras",
                                       compile=False)
    xs, predict = _model_predictor(dim, model, cfg["grid"])
    g = gaussian_factor(cfg["amplitude"], cfg["width"], dim)

    rows = []
    for t in TIMES:
        u_exact = exact_solution(dim, g, L, alpha, float(t), xs)
        u_nn = predict(float(t))
        err = u_nn - u_exact
        rows.append({
            "dim": dim,
            "t": float(t),
            "l2_rel": float(np.linalg.norm(err) / np.linalg.norm(u_exact)),
            "max_abs": float(np.max(np.abs(err))),
        })

    ic_ref = ic_field(dim, g, xs)
    ic_reconstruct = exact_solution(dim, g, L, alpha, 0.0, xs)
    series_check = float(np.max(np.abs(ic_ref - ic_reconstruct)))

    residuals = _residuals(dim, model, cfg)
    return rows, series_check, residuals, predict, xs, model


def _plot_1d(model, predict, xs, cfg):
    L, T, alpha = cfg["L"], cfg["T"], cfg["alpha"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")

    for t in [0.25, 0.5, 0.75, 1.0]:
        u = predict(float(t))
        axes[0].plot(xs, u, label=f"t = {t:.2f}s")

    g = gaussian_factor(cfg["amplitude"], cfg["width"], 1)
    for t in [0.25, 0.5, 0.75, 1.0]:
        axes[0].plot(xs, exact_solution(1, g, L, alpha, float(t), xs),
                     linestyle="--", color="gray", linewidth=0.8)
    axes[0].set_xlim(-L, L)
    axes[0].set_title("PINN prediction (solid) vs analytic (dashed)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("Temperature")
    axes[0].legend(fontsize=8)

    t_grid = np.linspace(0, T, 100)
    XT, TT = np.meshgrid(xs, t_grid)
    pts = np.stack([XT.ravel(), TT.ravel()], axis=1)
    u = model.predict(pts, verbose=0)[:, 0].reshape(XT.shape)
    u_exact = np.array([exact_solution(1, g, L, alpha, float(t), xs)
                        for t in t_grid])
    err = np.abs(u - u_exact)
    im = axes[1].pcolormesh(xs, t_grid, err, shading="auto", cmap="magma")
    axes[1].set_title("Pointwise |error| over (x, t)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("t")
    fig.colorbar(im, ax=axes[1])
    fig.savefig(OUT_DIR / "1d_error.png", dpi=110)
    plt.close(fig)


def _plot_2d_3d(dim, predict, xs, cfg):
    L, alpha = cfg["L"], cfg["alpha"]
    t_target = [0.5, 1.0]
    g = gaussian_factor(cfg["amplitude"], cfg["width"], dim)

    fig, axes = plt.subplots(1, len(t_target), figsize=(9, 4), layout="constrained")
    for ax, t in zip(axes, t_target):
        u_nn = predict(float(t))
        u_exact = exact_solution(dim, g, L, alpha, float(t), xs)
        if dim == 2:
            err = np.abs(u_nn - u_exact)
        else:
            err = np.abs(u_nn - u_exact)[:, :, xs.size // 2]  # mid-plane z = 0
        im = ax.imshow(err, cmap="magma", origin="lower",
                       extent=[-L, L, -L, L])
        ax.set_title(f"|error| on slice at t = {t:.2f}s")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax)
    fig.savefig(OUT_DIR / f"{dim}d_error.png", dpi=110)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    report = ["# PINN Validation Report", ""]
    report.append(
        "Trained models evaluated against the analytic heat-equation solution "
        "(sine series with zero Dirichlet BCs). No retraining involved.")
    report.append("")

    for dim in (1, 2, 3):
        rows, series_check, residuals, predict, xs, model = evaluate_dim(dim)
        cfg = CONFIGS[dim]
        all_rows.extend(rows)

        report.append(f"## {dim}D (L={cfg['L']}, T={cfg['T']}, "
                      f"alpha={cfg['alpha']}, IC amplitude={cfg['amplitude']}, "
                      f"width={cfg['width']})")
        report.append("")
        report.append("| t | L2 relative error | max abs error |")
        report.append("|---|---|---|")
        for r in rows:
            report.append(f"| {r['t']:.2f} | {r['l2_rel']:.3e} | {r['max_abs']:.3e} |")
        report.append("")
        report.append("| Physics residual | RMS value |")
        report.append("|---|---|")
        report.append(f"| PDE | {residuals['pde_rms']:.3e} |")
        report.append(f"| Boundary | {residuals['bc_rms']:.3e} |")
        report.append(f"| Initial condition | {residuals['ic_rms']:.3e} |")
        report.append("")
        report.append(f"Series self-check: max |IC - series(t=0)| = {series_check:.3e}")
        report.append("")

        if dim == 1:
            _plot_1d(model, predict, xs, cfg)
        else:
            _plot_2d_3d(dim, predict, xs, cfg)

    (OUT_DIR / "validation_report.md").write_text("\n".join(report), encoding="utf-8")

    import csv
    with open(OUT_DIR / "error_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dim", "t", "l2_rel", "max_abs"])
        writer.writeheader()
        writer.writerows(all_rows)

    print("\n".join(report))
    print(f"\nResults written to {OUT_DIR}")


if __name__ == "__main__":
    main()
