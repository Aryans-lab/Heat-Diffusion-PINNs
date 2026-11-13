"""T3 - Resolution dependence of the PINN prior in d = 1, 2, 3.

The L2 residual penalty corresponds to a stochastic heat equation driven by
space-time white noise.  Its point variance is

    s^2(Lambda) = sum_{|k| <= Lambda} phi_k(x)^2 (1 - exp(-2 a_k t)) / (2 a_k),   a_k = alpha pi^2 |k|^2,

which converges in d = 1, grows like log(Lambda) in d = 2 and like Lambda in
d = 3.  With point measurements the ideal-PINN estimator therefore has a
well-defined continuum limit only in one space dimension; in higher dimension
the answer depends on the resolution of the function class (for a network:
its spectral bias / training time).  Part (a) shows the growth of s^2, part
(b) shows the induced cut-off dependence of the 2D tree-level estimator at
fixed loss weights.
"""

from __future__ import annotations

import numpy as np

from ..exact import HeatBox, World, run_study, sensor_grid
from ..utils import COLORS, load_json, save_json, set_style

PRESETS = {
    "quick": dict(alpha0=0.1, cutoffs={1: [4, 8, 16, 32, 64, 128, 256], 2: [4, 8, 16, 32, 64],
                                       3: [4, 8, 12, 16]},
                  est_dim=2, est_cutoffs=[4, 8, 16], est_experiments=200, n_space=3, n_time=8,
                  sigma=0.02, weights_ratio=0.02, seed=20),
    "paper": dict(alpha0=0.1, cutoffs={1: [4, 8, 16, 32, 64, 128, 256, 512, 1024],
                                       2: [4, 8, 16, 32, 64, 128, 256], 3: [4, 8, 16, 24, 32, 48]},
                  est_dim=2, est_cutoffs=[4, 8, 16, 32, 64], est_experiments=2000, n_space=3,
                  n_time=8, sigma=0.02, weights_ratio=0.02, seed=20),
}


def run(cfg: dict, out) -> dict:
    a0 = cfg["alpha0"]
    var = {}
    for d, cuts in cfg["cutoffs"].items():
        d = int(d)
        X, t = sensor_grid(d, 3, 4)
        var[d] = []
        for L in cuts:
            p = HeatBox(dim=d, n_modes=L, cutoff="ball", ic_modes=4)
            var[d].append(float(p.kernel_diag(a0, X, t).mean()))
        print(f"[uv_divergence] d={d}: s^2(Lambda) = " + ", ".join(f"{v:.4f}" for v in var[d]))

    d = cfg["est_dim"]
    X, t = sensor_grid(d, cfg["n_space"], cfg["n_time"])
    n = len(t)
    est = []
    for L in cfg["est_cutoffs"]:
        p = HeatBox(dim=d, n_modes=L, cutoff="ball", ic_modes=min(8, L))
        pv = float(p.kernel_diag(a0, X, t).mean())
        rho = float(np.sqrt(pv * p.volume / (cfg["weights_ratio"] * n)))
        res = run_study(p, X, t, cfg["sigma"], a0, World("exact"), [rho],
                        n_experiments=cfg["est_experiments"], seed=cfg["seed"], n_grid=201)
        e = res["estimators"][f"tree@{rho:g}"]
        est.append(dict(cutoff=L, rho=rho, **e, mle_sd=res["estimators"]["mle"]["sd"]))
        print(f"    d={d} Lambda={L:4d}: rho={rho:.3f} tree bias={e['bias']:+.2f} sd={e['sd']:.2f} "
              f"edge={e['edge_frac']:.2f}")
    save_json(out / "results.json", {"config": cfg, "variance": var, "estimator": est})
    return {"variance": var, "estimator": est}


def plot(out) -> None:
    set_style()
    import matplotlib.pyplot as plt

    data = load_json(out / "results.json")
    cfg = data["config"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    for d, c in zip(("1", "2", "3"), ("#0072B2", "#D55E00", "#009E73")):
        cuts = np.array(cfg["cutoffs"][d])
        v = np.array(data["variance"][d])
        axes[0].plot(cuts, v / v[0], "o-", ms=3, color=c, label=f"d = {d}")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("spectral cut-off $\\Lambda$")
    axes[0].set_ylabel("prior variance $s^2(\\Lambda)\\,/\\,s^2(\\Lambda_0)$")
    axes[0].legend()
    est = data["estimator"]
    L = [e["cutoff"] for e in est]
    axes[1].plot(L, [e["rho"] for e in est], "o-", color=COLORS["tree"])
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(L)
    axes[1].set_xticklabels([f"{int(v)}" for v in L])
    axes[1].minorticks_off()
    axes[1].set_xlabel("spectral cut-off $\\Lambda$")
    axes[1].set_ylabel("effective $\\rho$ at fixed loss weights")
    axes[2].errorbar(L, [e["bias"] for e in est], yerr=[e["bias_err"] for e in est], fmt="o-",
                     color=COLORS["tree"], label="bias / CRB")
    axes[2].plot(L, [e["sd"] for e in est], "s--", color=COLORS["oneloop"], label="std. dev. / CRB")
    axes[2].set_xscale("log", base=2)
    axes[2].set_xticks(L)
    axes[2].set_xticklabels([f"{int(v)}" for v in L])
    axes[2].minorticks_off()
    axes[2].set_xlabel("spectral cut-off $\\Lambda$")
    axes[2].set_title(f"{cfg['est_dim']}D tree-level estimator, fixed weights", fontsize=9)
    axes[2].legend()
    fig.savefig(out / "uv_divergence.png")
    plt.close(fig)
