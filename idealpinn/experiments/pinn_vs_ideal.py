"""N1 - Does a real PINN converge to the ideal-PINN (tree-level) estimator?

For a handful of noisy data sets and several loss-weight ratios the script
trains actual inverse PINNs (hard-constrained initial/boundary conditions)
with Levenberg-Marquardt and with Adam, records the trajectory of the inferred
diffusivity, and compares the end point with two numbers computed in closed
form for the *same* data: the hard-constrained MLE and the tree-level
estimate.  The theory predicts that the PINN goes to the latter.
"""

from __future__ import annotations

import numpy as np
import torch

from ..nn import AdamConfig, HeatPINN, Weights, adam_fit
from ..utils import COLORS, load_json, save_json, set_style
from ._neural import (dense_residual, exact_reference, inverse_setup, lm_in_groups, member_layout,
                      net_spec)

PRESETS = {
    "quick": dict(alpha0=0.1, sigma=0.02, n_space=5, n_time=20, n_datasets=1, seed=3, n_modes=128,
                  rhos=[0.1, 0.3, 1.0], width=24, depth=3, n_colloc=2048, alpha_init=0.05,
                  archs=["mlp"], lm_iters=200, adam_steps=3000, adam_lr=2e-3, group_size=8,
                  init_seed=0, alpha_warmup=40),
    "paper": dict(alpha0=0.1, sigma=0.02, n_space=5, n_time=20, n_datasets=4, seed=3, n_modes=160,
                  rhos=[0.1, 0.3, 1.0, 2.0], width=24, depth=3, n_colloc=4096, alpha_init=0.05,
                  archs=["mlp", "fourier"], lm_iters=800, adam_steps=60000, adam_lr=2e-3,
                  group_size=8, init_seed=0, alpha_warmup=40),
}


def run(cfg: dict, out) -> dict:
    s = inverse_setup(cfg)
    ref = exact_reference(s, cfg["rhos"])
    d_idx, r_idx = member_layout(cfg["n_datasets"], ref["ratios"])
    E = len(d_idx)
    result = dict(config=cfg, crb=s["crb"], mle=ref["mle"][d_idx], tree=ref["tree"][r_idx, d_idx],
                  ideal_loss=ref["ideal_loss"][r_idx, d_idx], rho=np.array(cfg["rhos"])[r_idx],
                  dataset=d_idx, ratios=ref["ratios"], runs={})
    for arch in cfg["archs"]:
        spec = net_spec(cfg, 2, arch == "fourier")
        pinn = HeatPINN(s["problem"], spec, constraint="ic_bc", n_colloc=cfg["n_colloc"],
                        sensors=(s["X"], s["t"]), learn_alpha=True, alpha_init=cfg["alpha_init"])
        weights = [Weights(data=1.0, res=ref["ratios"][r]) for r in r_idx]
        args = pinn.pack(E, Y=s["Y"][d_idx], weights=weights)
        P0 = pinn.init(E, cfg["init_seed"])
        print(f"[pinn_vs_ideal] arch={arch}  device={pinn.device}  networks={E}")

        P, loss_h, alpha_h, wall = lm_in_groups(pinn, P0, args, cfg["lm_iters"], cfg["group_size"],
                                            alpha_warmup=cfg["alpha_warmup"])
        terms = [v.cpu().numpy() for v in pinn.term_losses(P, args[0], args[1])]
        result["runs"][f"{arch}/lm"] = dict(
            alpha=alpha_h, loss=loss_h, seconds=wall, final_alpha=alpha_h[-1],
            mse_data=terms[0], mse_res=terms[1], mse_res_dense=dense_residual(pinn, P, args))

        if cfg["adam_steps"] > 0:
            traj, steps = [], []
            acfg = AdamConfig(steps=cfg["adam_steps"], lr=cfg["adam_lr"],
                              decay_steps=max(cfg["adam_steps"] // 10, 1), log_every=max(cfg["adam_steps"] // 300, 1))
            cb = lambda step, p, L: (traj.append(pinn.alpha_of(p).cpu().numpy().copy()), steps.append(step))
            P, hist = adam_fit(pinn, P0, args, acfg, callback=cb, verbose=True)
            terms = [v.cpu().numpy() for v in pinn.term_losses(P, args[0], args[1])]
            result["runs"][f"{arch}/adam"] = dict(
                alpha=np.stack(traj), steps=steps, seconds=hist["time"][-1], final_alpha=traj[-1],
                mse_data=terms[0], mse_res=terms[1], mse_res_dense=dense_residual(pinn, P, args))

        for name, r in result["runs"].items():
            if not name.startswith(arch):
                continue
            dev_tree = (r["final_alpha"] - result["tree"]) / s["crb"]
            dev_mle = (r["final_alpha"] - result["mle"]) / s["crb"]
            for rho in cfg["rhos"]:
                m = result["rho"] == rho
                print(f"    {name:14s} rho={rho:<4g}: (PINN - tree)/CRB = {dev_tree[m].mean():+.2f}"
                      f"   (PINN - MLE)/CRB = {dev_mle[m].mean():+.2f}"
                      f"   theory (tree - MLE)/CRB = {((result['tree'] - result['mle'])[m] / s['crb']).mean():+.2f}")
    save_json(out / "results.json", result)
    return result


def plot(out) -> None:
    set_style()
    import matplotlib.pyplot as plt

    r = load_json(out / "results.json")
    cfg, crb = r["config"], r["crb"]
    rho, ds = np.array(r["rho"]), np.array(r["dataset"])
    tree, mle = np.array(r["tree"]), np.array(r["mle"])
    cols = plt.cm.viridis(np.linspace(0.1, 0.9, len(cfg["rhos"])))
    runs = list(r["runs"].items())
    fig, axes = plt.subplots(1, len(runs), figsize=(4.6 * len(runs), 3.6), squeeze=False)
    for ax, (name, run) in zip(axes[0], runs):
        A = np.array(run["alpha"])
        x = np.arange(A.shape[0]) if "steps" not in run else np.array(run["steps"])
        for e in np.where(ds == 0)[0]:
            c = cols[cfg["rhos"].index(rho[e])]
            ax.plot(x, (A[:, e] - mle[e]) / crb, color=c, label=f"PINN, $\\rho$={rho[e]:g}")
            ax.axhline((tree[e] - mle[e]) / crb, color=c, ls="--", lw=1.0)
        ax.axhline(0.0, color=COLORS["mle"], lw=1.0)
        ax.set_yscale("symlog", linthresh=4.0)
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        ax.set_xlabel("iteration" if name.endswith("lm") else "Adam step")
        ax.set_ylabel("$(\\alpha - \\hat\\alpha_{\\rm MLE})\\,/\\,$CRB")
        ax.set_title(f"{name}  ({run['seconds']:.0f} s); dashed: exact tree-level prediction", fontsize=8)
        ax.legend(fontsize=7)
    fig.savefig(out / "pinn_vs_ideal.png")
    plt.close(fig)
