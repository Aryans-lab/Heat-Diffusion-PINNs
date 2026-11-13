"""T1 - Frequentist properties of the ideal inverse PINN versus the physics weight.

Question: if a PINN were trained perfectly, how good an estimator of the
diffusivity would it be, as a function of how strongly the PDE residual is
weighted against the data?

For every measurement design and every assumed model-error-to-noise ratio rho
(equivalently, loss-weight ratio w_res/w_data) the script computes, from
thousands of pseudo-experiments and without any neural network, the bias,
spread and interval coverage of

  tree      - the ideal PINN (joint minimisation over field and parameter),
  oneloop   - tree + fluctuation determinant (marginal likelihood, fixed rho),
  profiled  - data misfit of the soft-constrained field (bilevel outer loss),
  mle       - hard-constrained least squares,
  learned   - one-loop with the physics precision estimated from the data.
"""

from __future__ import annotations

import numpy as np

from ..exact import HeatBox, World, run_study, sensor_grid
from ..utils import COLORS, load_json, save_json, set_style

PRESETS = {
    "quick": dict(alpha0=0.1, n_modes=96, n_experiments=400, seed=0,
                  designs=[dict(n_space=5, n_time=20, sigma=0.02)],
                  rhos=list(np.round(np.logspace(-1.5, 0.5, 9), 4))),
    "paper": dict(alpha0=0.1, n_modes=160, n_experiments=4000, seed=0,
                  designs=[dict(n_space=5, n_time=20, sigma=0.02),
                           dict(n_space=5, n_time=20, sigma=0.05),
                           dict(n_space=3, n_time=10, sigma=0.02),
                           dict(n_space=9, n_time=40, sigma=0.02)],
                  rhos=list(np.round(np.logspace(-1.5, 0.5, 13), 4))),
}


def run(cfg: dict, out) -> dict:
    problem = HeatBox(dim=1, n_modes=cfg["n_modes"])
    results = []
    for i, d in enumerate(cfg["designs"]):
        X, t = sensor_grid(1, d["n_space"], d["n_time"], problem.T)
        res = run_study(problem, X, t, d["sigma"], cfg["alpha0"], World("exact"), cfg["rhos"],
                        n_experiments=cfg["n_experiments"], seed=cfg["seed"] + i)
        res["design"] = d
        results.append(res)
        m = res["meta"]
        print(f"[bias_vs_weight] design {d}: N={m['n_data']}  CRB/alpha0={m['crb'] / cfg['alpha0']:.3%}")
        for r in cfg["rhos"]:
            e = res["estimators"][f"tree@{r:g}"]
            print(f"    rho={r:<7g} tree: bias={e['bias']:+6.2f} sd={e['sd']:5.2f} cover={e['coverage']:.2f}"
                  f" edge={e['edge_frac']:.2f}")
    save_json(out / "results.json", {"config": cfg, "results": results})
    return {"results": results}


def plot(out) -> None:
    set_style()
    import matplotlib.pyplot as plt

    data = load_json(out / "results.json")
    rhos = np.array(data["config"]["rhos"])
    for i, res in enumerate(data["results"]):
        est, meta = res["estimators"], res["meta"]
        fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.3))
        panels = [("bias", "bias / CRB"), ("sd", "std. dev. / CRB"), ("rmse", "RMSE / CRB"),
                  ("coverage", "coverage of nominal 68% interval")]
        for ax, (key, label) in zip(axes, panels):
            for fam, name in [("tree", "tree level (ideal PINN)"), ("oneloop", "one loop, fixed $\\rho$"),
                              ("profiled", "profiled (bilevel outer)")]:
                y = np.array([est[f"{fam}@{r:g}"][key] for r in rhos])
                ok = np.array([est[f"{fam}@{r:g}"]["edge_frac"] for r in rhos]) < 0.02
                ax.plot(rhos[ok], y[ok], "o-", ms=3, color=COLORS[fam], label=name)
                if (~ok).any():                      # estimates escaping the scan range: lower bounds
                    ax.plot(rhos[~ok], y[~ok], "x", ms=4, color=COLORS[fam], alpha=0.5)
            ax.axhline(est["mle"][key], color=COLORS["mle"], ls="--", label="MLE (hard constraint)")
            ax.axhline(est["oneloop_learned"][key], color=COLORS["learned"], ls=":",
                       label="one loop, learned $\\rho$")
            ax.set_xscale("log")
            ax.set_xlabel("assumed model-error / noise ratio $\\rho$")
            ax.set_ylabel(label)
            if key == "coverage":
                ax.axhline(0.6827, color=COLORS["grey"], lw=0.8)
                ax.set_ylim(0, 1.02)
            elif key == "bias":
                ax.set_yscale("symlog", linthresh=1.0)
            else:
                ax.set_yscale("log")
            rho_vanilla = np.sqrt(meta["prior_var_unit"] * 1.0 / meta["n_data"])
            ax.axvline(rho_vanilla, color=COLORS["grey"], lw=0.8, ls="-.")
        axes[0].legend(fontsize=7, loc="lower left")
        d = res["design"]
        fig.suptitle(f"1D heat equation, {d['n_space']} sensors x {d['n_time']} times (N={meta['n_data']}), "
                     f"$\\sigma$={d['sigma']}; dash-dotted: equal loss weights", fontsize=10)
        fig.savefig(out / f"bias_vs_weight_design{i}.png")
        plt.close(fig)
