"""N3 - Loss balancing silently changes the estimator.

Adam-trained inverse PINNs with (i) equal weights, (ii) a hand-tuned
"emphasise the data" weight and (iii) gradient-norm balancing.  For every run
the effective weight ratio w_res/w_data is converted into the model-error
ratio rho it implies, and the inferred diffusivity is compared with the
closed-form tree-level prediction *at that ratio*.
"""

from __future__ import annotations

import numpy as np

from ..nn import AdamConfig, HeatPINN, Weights, adam_fit
from ..utils import COLORS, load_json, save_json, set_style
from ._neural import inverse_setup, net_spec, tree_estimate_at_ratio

PRESETS = {
    "quick": dict(alpha0=0.1, sigma=0.02, n_space=5, n_time=20, n_datasets=2, seed=11, n_modes=128,
                  width=24, depth=3, n_colloc=2048, alpha_init=0.07, adam_steps=3000, adam_lr=2e-3,
                  schemes=["equal", "data_x100", "gradnorm"], init_seed=0),
    "paper": dict(alpha0=0.1, sigma=0.02, n_space=5, n_time=20, n_datasets=16, seed=11, n_modes=160,
                  width=32, depth=3, n_colloc=4096, alpha_init=0.07, adam_steps=80000, adam_lr=2e-3,
                  schemes=["equal", "data_x10", "data_x100", "data_x1000", "gradnorm"], init_seed=0),
}


def _scheme(name: str) -> tuple[Weights, str]:
    if name == "equal":
        return Weights(data=1.0, res=1.0), "fixed"
    if name.startswith("data_x"):
        return Weights(data=float(name[6:]), res=1.0), "fixed"
    if name == "gradnorm":
        return Weights(data=1.0, res=1.0), "gradnorm"
    raise ValueError(name)


def run(cfg: dict, out) -> dict:
    s = inverse_setup(cfg)
    E = cfg["n_datasets"]
    spec = net_spec(cfg, 2, False)
    res = dict(config=cfg, crb=s["crb"], runs={})
    from ..exact import argmin_refined, objective_curves
    grid = np.linspace(s["alpha0"] - 16 * s["crb"], s["alpha0"] + 16 * s["crb"], 321)
    mle = argmin_refined(grid, objective_curves(s["problem"], s["X"], s["t"], s["Y"], s["sigma"], grid,
                                                [1.0]).mle)[0]
    res["mle"] = mle
    for name in cfg["schemes"]:
        w, mode = _scheme(name)
        pinn = HeatPINN(s["problem"], spec, constraint="ic_bc", n_colloc=cfg["n_colloc"],
                        sensors=(s["X"], s["t"]), learn_alpha=True, alpha_init=cfg["alpha_init"])
        args = pinn.pack(E, Y=s["Y"], weights=w)
        acfg = AdamConfig(steps=cfg["adam_steps"], lr=cfg["adam_lr"], weighting=mode,
                          decay_steps=max(cfg["adam_steps"] // 10, 1),
                          log_every=max(cfg["adam_steps"] // 300, 1))
        traj, steps = [], []
        cb = lambda step, p, L: (traj.append(pinn.alpha_of(p).cpu().numpy().copy()), steps.append(step))
        print(f"[adaptive_weights] scheme={name} device={pinn.device}")
        P, hist = adam_fit(pinn, pinn.init(E, cfg["init_seed"]), args, acfg, callback=cb, verbose=True)
        W = hist["weights"].numpy()                                    # (T, E, terms)
        ratio = W[:, :, 1] / W[:, :, 0]
        rho = np.sqrt(s["pv"] * s["problem"].volume / (ratio * s["n"]))
        final = traj[-1]
        target = np.array([tree_estimate_at_ratio(s, s["Y"][e], float(ratio[-1, e])) for e in range(E)])
        res["runs"][name] = dict(steps=steps, alpha=np.stack(traj), ratio=ratio, rho=rho,
                                 final_alpha=final, tree_at_final_ratio=target,
                                 seconds=hist["time"][-1], terms=hist["terms"].numpy()[-1])
        print(f"    final rho_eff (median) = {np.median(rho[-1]):.3f};  (PINN - MLE)/CRB = "
              f"{np.mean(final - mle) / s['crb']:+.2f};  tree-level prediction at that rho: "
              f"{np.mean(target - mle) / s['crb']:+.2f}")
    save_json(out / "results.json", res)
    return res


def plot(out) -> None:
    set_style()
    import matplotlib.pyplot as plt

    r = load_json(out / "results.json")
    crb, mle = r["crb"], np.array(r["mle"])
    names = list(r["runs"])
    cols = plt.cm.plasma(np.linspace(0.05, 0.85, len(names)))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for c, name in zip(cols, names):
        run = r["runs"][name]
        x = np.array(run["steps"]) + 1
        axes[0].plot(x, np.median(np.array(run["rho"]), axis=1), color=c, label=name)
        A = np.array(run["alpha"])
        axes[1].plot(x, ((A - mle[None]) / crb).mean(1), color=c, label=name)
        axes[1].axhline(np.mean(np.array(run["tree_at_final_ratio"]) - mle) / crb, color=c, ls="--", lw=1)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Adam step")
    axes[0].set_ylabel("implied model-error ratio $\\rho$")
    axes[0].legend(fontsize=7)
    axes[1].set_xscale("log")
    axes[1].set_yscale("symlog", linthresh=2.0)
    axes[1].set_xlabel("Adam step")
    axes[1].set_ylabel("mean $(\\alpha-\\hat\\alpha_{\\rm MLE})/$CRB")
    axes[1].set_title("dashed: tree-level prediction at the final weights", fontsize=8)
    fig.savefig(out / "adaptive_weights.png")
    plt.close(fig)
