"""T2 - Which estimator survives when the physics in the loss is imperfect?

Three worlds generate the pseudo-data; every estimator always assumes the
pure heat equation:

  exact       the model is right,
  stochastic  the true dynamics has random model error  u_t = a u_xx + xi,
  loss        the true dynamics has an unmodelled cooling term  -h u.
"""

from __future__ import annotations

import numpy as np

from ..exact import HeatBox, World, run_study, sensor_grid
from ..utils import COLORS, load_json, save_json, set_style

PRESETS = {
    "quick": dict(alpha0=0.1, n_modes=96, n_experiments=400, seed=10, n_space=5, n_time=20,
                  sigma=0.02, rhos=[0.1, 0.3, 1.0],
                  worlds=[dict(kind="exact"), dict(kind="stochastic", rho_true=1.0),
                          dict(kind="loss", loss_rate=0.05)]),
    "paper": dict(alpha0=0.1, n_modes=160, n_experiments=4000, seed=10, n_space=5, n_time=20,
                  sigma=0.02, rhos=[0.1, 0.3, 1.0, 3.0],
                  worlds=[dict(kind="exact"),
                          dict(kind="stochastic", rho_true=0.3),
                          dict(kind="stochastic", rho_true=1.0),
                          dict(kind="stochastic", rho_true=3.0),
                          dict(kind="loss", loss_rate=0.02),
                          dict(kind="loss", loss_rate=0.05)]),
}


def _label(w: dict) -> str:
    if w["kind"] == "exact":
        return "exact physics"
    if w["kind"] == "stochastic":
        return f"random model error, rho_true={w['rho_true']:g}"
    return f"unmodelled cooling, h={w['loss_rate']:g}"


def run(cfg: dict, out) -> dict:
    problem = HeatBox(dim=1, n_modes=cfg["n_modes"])
    X, t = sensor_grid(1, cfg["n_space"], cfg["n_time"], problem.T)
    results = []
    for i, w in enumerate(cfg["worlds"]):
        res = run_study(problem, X, t, cfg["sigma"], cfg["alpha0"], World(**w), cfg["rhos"],
                        n_experiments=cfg["n_experiments"], seed=cfg["seed"] + i, span=30.0, n_grid=401)
        res["label"] = _label(w)
        results.append(res)
        print(f"[misspecification] {res['label']}  (CRB of this world / exact CRB = "
              f"{res['meta']['crb_world'] / res['meta']['crb']:.2f})")
        for name, e in res["estimators"].items():
            if name.startswith("profiled"):
                continue
            print(f"    {name:16s} bias={e['bias']:+7.2f} sd={e['sd']:6.2f} rmse={e['rmse']:6.2f} "
                  f"cover={e['coverage']:.2f} edge={e['edge_frac']:.2f}")
    save_json(out / "results.json", {"config": cfg, "results": results})
    _table(out, results, cfg)
    return {"results": results}


def _table(out, results, cfg) -> None:
    names = ["mle"] + [f"tree@{r:g}" for r in cfg["rhos"]] + [f"oneloop@{r:g}" for r in cfg["rhos"]] \
            + ["oneloop_learned"]
    lines = ["| world | estimator | bias/CRB | sd/CRB | RMSE/CRB | 68% coverage |", "|---|---|---:|---:|---:|---:|"]
    for res in results:
        for n in names:
            e = res["estimators"][n]
            flag = " (scan edge)" if e["edge_frac"] > 0.02 else ""
            lines.append(f"| {res['label']} | {n}{flag} | {e['bias']:+.2f} | {e['sd']:.2f} | "
                         f"{e['rmse']:.2f} | {e['coverage']:.2f} |")
    (out / "table.md").write_text("\n".join(lines) + "\n")


def plot(out) -> None:
    set_style()
    import matplotlib.pyplot as plt

    data = load_json(out / "results.json")
    cfg, results = data["config"], data["results"]
    names = ["mle"] + [f"tree@{r:g}" for r in cfg["rhos"]] + [f"oneloop@{r:g}" for r in cfg["rhos"]] \
            + ["oneloop_learned"]
    cols = [COLORS["mle"]] + [COLORS["tree"]] * len(cfg["rhos"]) + [COLORS["oneloop"]] * len(cfg["rhos"]) \
           + [COLORS["learned"]]
    fig, axes = plt.subplots(2, len(results), figsize=(3.4 * len(results), 5.6), squeeze=False)
    x = np.arange(len(names))
    for j, res in enumerate(results):
        rm = [res["estimators"][n]["rmse"] for n in names]
        cv = [res["estimators"][n]["coverage"] for n in names]
        axes[0, j].bar(x, rm, color=cols)
        axes[0, j].set_yscale("log")
        axes[0, j].axhline(res["meta"]["crb_world"] / res["meta"]["crb"], color=COLORS["grey"], lw=0.8)
        axes[0, j].set_title(res["label"], fontsize=9)
        axes[1, j].bar(x, cv, color=cols)
        axes[1, j].axhline(0.6827, color=COLORS["grey"], lw=0.8)
        axes[1, j].set_ylim(0, 1)
        for ax in axes[:, j]:
            ax.set_xticks(x)
            ax.set_xticklabels([n.replace("oneloop", "1loop").replace("_learned", "\nlearned") for n in names],
                               rotation=90, fontsize=7)
    axes[0, 0].set_ylabel("RMSE / CRB(exact world)")
    axes[1, 0].set_ylabel("coverage of nominal 68% interval")
    fig.savefig(out / "misspecification.png")
    plt.close(fig)
