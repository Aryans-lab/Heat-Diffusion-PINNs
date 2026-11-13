"""F1 - Forward heat equation in 1D/2D/3D: first-year recipe vs. current practice.

Three training recipes on the same problem, scored against the exact solution:

  legacy  soft initial/boundary conditions, plain tanh MLP, Adam, equal weights
          (the approach of the original version of this repository);
  modern  exact initial/boundary conditions, Fourier features, random weight
          factorisation, causal weighting, Adam with learning-rate decay;
  lm      exact initial/boundary conditions + Levenberg-Marquardt (Gauss-Newton).

Reported: relative L2 error (mean and spread over seeds), wall time, parameters.
"""

from __future__ import annotations

import time

import numpy as np

from ..exact import HeatBox
from ..nn import AdamConfig, HeatPINN, NetSpec, adam_fit
from ..utils import load_json, save_json, set_style
from ._neural import lm_in_groups

PRESETS = {
    "quick": dict(alpha0=0.1, dims=[1], n_seeds=2, recipes=["legacy", "modern", "lm"],
                  width=24, depth=3, n_colloc=2048, adam_steps=3000, lm_iters=150, group_size=4),
    "paper": dict(alpha0=0.1, dims=[1, 2, 3], n_seeds=5, recipes=["legacy", "modern", "lm"],
                  width=32, depth=3, n_colloc=4096, adam_steps=40000, lm_iters=500, group_size=5),
}


def run(cfg: dict, out) -> dict:
    rows = []
    for d in cfg["dims"]:
        problem = HeatBox(dim=d, n_modes=16)                      # kernel not needed here
        for recipe in cfg["recipes"]:
            fourier = recipe == "modern"
            spec = NetSpec(in_dim=d + 1, width=cfg["width"], depth=cfg["depth"],
                           fourier_features=16 if fourier else 0, fourier_scale=3.0, rwf=fourier)
            pinn = HeatPINN(problem, spec, constraint="none" if recipe == "legacy" else "ic_bc",
                            n_colloc=cfg["n_colloc"], n_ic=cfg["n_colloc"] // 4, n_bc=cfg["n_colloc"] // 4)
            E = cfg["n_seeds"]
            P0 = pinn.init(E, seed=100 + d)
            args = pinn.pack(E, alpha=cfg["alpha0"])
            print(f"[forward_benchmark] d={d} recipe={recipe} device={pinn.device} params={pinn.n_params}")
            if recipe == "lm":
                P, loss_h, _, wall = lm_in_groups(pinn, P0, args, cfg["lm_iters"], cfg["group_size"])
            else:
                acfg = AdamConfig(steps=cfg["adam_steps"], lr=1e-3,
                                  decay_rate=1.0 if recipe == "legacy" else 0.9,
                                  decay_steps=max(cfg["adam_steps"] // 20, 1), resample_every=100,
                                  causal_eps=1.0 if recipe == "modern" else 0.0,
                                  log_every=max(cfg["adam_steps"] // 100, 1))
                t0 = time.time()
                P, hist = adam_fit(pinn, P0, args, acfg, verbose=True)
                wall = time.time() - t0
            err = pinn.field_error(P, cfg["alpha0"])
            row = dict(dim=d, recipe=recipe, params=pinn.n_params, seconds=wall,
                       l2_rel=err["l2_rel"], max_abs=err["max_abs"],
                       l2_mean=float(np.mean(err["l2_rel"])), l2_min=float(np.min(err["l2_rel"])),
                       l2_max=float(np.max(err["l2_rel"])))
            rows.append(row)
            print(f"    relative L2 error: mean {row['l2_mean']:.2e}  [min {row['l2_min']:.2e}, "
                  f"max {row['l2_max']:.2e}] over {E} seeds;  {wall:.0f} s")
    save_json(out / "results.json", {"config": cfg, "rows": rows})
    lines = ["| d | recipe | parameters | rel. L2 error (mean) | min | max | seconds (all seeds) |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['dim']} | {r['recipe']} | {r['params']} | {r['l2_mean']:.2e} | "
                     f"{r['l2_min']:.2e} | {r['l2_max']:.2e} | {r['seconds']:.0f} |")
    (out / "table.md").write_text("\n".join(lines) + "\n")
    return {"rows": rows}


def plot(out) -> None:
    set_style()
    import matplotlib.pyplot as plt

    data = load_json(out / "results.json")
    rows, cfg = data["rows"], data["config"]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    width = 0.25
    cols = {"legacy": "#999999", "modern": "#56B4E9", "lm": "#D55E00"}
    for i, rec in enumerate(cfg["recipes"]):
        rr = [r for r in rows if r["recipe"] == rec]
        x = np.arange(len(rr)) + (i - 1) * width
        m = np.array([r["l2_mean"] for r in rr])
        lo = m - np.array([r["l2_min"] for r in rr])
        hi = np.array([r["l2_max"] for r in rr]) - m
        ax.bar(x, m, width, yerr=[lo, hi], color=cols[rec], label=rec, capsize=2)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(cfg["dims"])))
    ax.set_xticklabels([f"{d}D" for d in cfg["dims"]])
    ax.set_ylabel("relative $L^2$ error vs exact solution")
    ax.legend()
    fig.savefig(out / "forward_benchmark.png")
    plt.close(fig)
