"""N2 - Ensemble test: do trained PINNs reproduce the predicted bias and spread?

M independent noisy data sets x several loss-weight ratios are fitted by
real inverse PINNs (Levenberg-Marquardt, all networks in parallel).  The
resulting estimates are compared data set by data set with the closed-form
tree-level estimates, and their ensemble bias / spread with the exact curves.
"""

from __future__ import annotations

import numpy as np

from ..exact import World, run_study
from ..nn import HeatPINN, Weights
from ..utils import COLORS, load_json, save_json, set_style
from ._neural import exact_reference, inverse_setup, lm_in_groups, member_layout, net_spec

PRESETS = {
    "quick": dict(alpha0=0.1, sigma=0.02, n_space=5, n_time=20, n_datasets=4, seed=7, n_modes=128,
                  rhos=[0.1, 1.0], width=20, depth=3, n_colloc=1024, alpha_init=0.07,
                  lm_iters=250, group_size=8, init_seed=0, alpha_warmup=40, certify_tol=0.01,
                  theory_experiments=1000),
    "paper": dict(alpha0=0.1, sigma=0.02, n_space=5, n_time=20, n_datasets=64, seed=7, n_modes=160,
                  rhos=[0.1, 0.5, 1.0], width=24, depth=3, n_colloc=2048, alpha_init=0.07,
                  lm_iters=600, group_size=8, init_seed=0, alpha_warmup=40, certify_tol=0.01,
                  theory_experiments=4000),
}


def run(cfg: dict, out) -> dict:
    s = inverse_setup(cfg)
    ref = exact_reference(s, cfg["rhos"])
    d_idx, r_idx = member_layout(cfg["n_datasets"], ref["ratios"])
    E = len(d_idx)
    spec = net_spec(cfg, 2, False)
    pinn = HeatPINN(s["problem"], spec, constraint="ic_bc", n_colloc=cfg["n_colloc"],
                    sensors=(s["X"], s["t"]), learn_alpha=True, alpha_init=cfg["alpha_init"])
    args = pinn.pack(E, Y=s["Y"][d_idx], weights=[Weights(data=1.0, res=ref["ratios"][r]) for r in r_idx])
    print(f"[pinn_ensemble] device={pinn.device} networks={E}")
    P, loss_h, alpha_h, wall = lm_in_groups(pinn, pinn.init(E, cfg["init_seed"]), args,
                                            cfg["lm_iters"], cfg["group_size"],
                                            alpha_warmup=cfg["alpha_warmup"])
    theory = run_study(s["problem"], s["X"], s["t"], s["sigma"], s["alpha0"], World("exact"),
                       cfg["rhos"], n_experiments=cfg["theory_experiments"], seed=cfg["seed"] + 1)
    crb, a0 = s["crb"], s["alpha0"]
    pinn_a = alpha_h[-1]
    tree_a, mle_a = ref["tree"][r_idx, d_idx], ref["mle"][d_idx]
    summary = []
    M = cfg["n_datasets"]
    ideal = ref["ideal_loss"][r_idx, d_idx]
    # The closed form gives the value the loss must reach, so training can be certified:
    # above it the optimiser has not converged; below it the network has exploited the
    # finite collocation set (the continuum problem cannot go below the ideal value).
    loss_ratio = loss_h[-1] / ideal
    certified = np.abs(loss_ratio - 1.0) <= cfg["certify_tol"]
    for k, rho in enumerate(cfg["rhos"]):
        m = r_idx == k
        c = m & certified
        th = theory["estimators"][f"tree@{rho:g}"]
        nc = int(c.sum())
        row = dict(rho=rho, n=int(m.sum()), n_certified=nc,
                   pinn_bias=float((pinn_a[m].mean() - a0) / crb),
                   pinn_bias_err=float(pinn_a[m].std(ddof=1) / np.sqrt(M) / crb),
                   pinn_sd=float(pinn_a[m].std(ddof=1) / crb),
                   tree_bias_same_data=float((tree_a[m].mean() - a0) / crb),
                   mle_bias_same_data=float((mle_a[m].mean() - a0) / crb),
                   theory_bias=th["bias"], theory_sd=th["sd"],
                   rms_pinn_minus_tree=float(np.sqrt(np.mean((pinn_a[m] - tree_a[m]) ** 2)) / crb),
                   rms_tree_minus_mle=float(np.sqrt(np.mean((tree_a[m] - mle_a[m]) ** 2)) / crb),
                   loss_ratio=float(np.median(loss_h[-1][m] / ideal[m])))
        if nc > 1:
            row.update(cert_pinn_bias=float((pinn_a[c].mean() - a0) / crb),
                       cert_pinn_bias_err=float(pinn_a[c].std(ddof=1) / np.sqrt(nc) / crb),
                       cert_pinn_sd=float(pinn_a[c].std(ddof=1) / crb),
                       cert_tree_bias_same_data=float((tree_a[c].mean() - a0) / crb),
                       cert_rms_pinn_minus_tree=float(np.sqrt(np.mean((pinn_a[c] - tree_a[c]) ** 2)) / crb),
                       cert_rms_pinn_minus_mle=float(np.sqrt(np.mean((pinn_a[c] - mle_a[c]) ** 2)) / crb))
        summary.append(row)
        print(f"    rho={rho:<4g} all {row['n']} networks: PINN bias={row['pinn_bias']:+.2f}+-{row['pinn_bias_err']:.2f} "
              f"(tree, same data: {row['tree_bias_same_data']:+.2f}; theory: {th['bias']:+.2f})  "
              f"sd={row['pinn_sd']:.2f} (theory {th['sd']:.2f})  "
              f"rms(PINN-tree)={row['rms_pinn_minus_tree']:.2f} CRB  loss/ideal={row['loss_ratio']:.4f}")
        if nc > 1:
            print(f"             {nc} certified (|loss/ideal - 1| <= {cfg['certify_tol']}): "
                  f"PINN bias={row['cert_pinn_bias']:+.2f}+-{row['cert_pinn_bias_err']:.2f} "
                  f"(tree, same data: {row['cert_tree_bias_same_data']:+.2f})  sd={row['cert_pinn_sd']:.2f}  "
                  f"rms(PINN-tree)={row['cert_rms_pinn_minus_tree']:.2f} CRB  "
                  f"rms(PINN-MLE)={row['cert_rms_pinn_minus_mle']:.2f} CRB")
    res = dict(config=cfg, crb=crb, seconds=wall, pinn=pinn_a, tree=tree_a, mle=mle_a,
               rho=np.array(cfg["rhos"])[r_idx], dataset=d_idx, final_loss=loss_h[-1],
               ideal_loss=ideal, loss_ratio=loss_ratio, certified=certified, summary=summary,
               theory={k: v for k, v in theory["estimators"].items()})
    save_json(out / "results.json", res)
    return res


def plot(out) -> None:
    set_style()
    import matplotlib.pyplot as plt

    r = load_json(out / "results.json")
    cfg, crb, a0 = r["config"], r["crb"], r["config"]["alpha0"]
    rho = np.array(r["rho"])
    pinn, tree, mle = (np.array(r[k]) for k in ("pinn", "tree", "mle"))
    cols = plt.cm.viridis(np.linspace(0.1, 0.9, len(cfg["rhos"])))
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7))
    cert = np.array(r["certified"], dtype=bool)
    for c, rv in zip(cols, cfg["rhos"]):
        m = (rho == rv) & cert
        axes[0].plot((tree[m] - a0) / crb, (pinn[m] - a0) / crb, "o", ms=3, color=c, label=f"$\\rho$={rv:g}")
        axes[1].plot((mle[m] - a0) / crb, (pinn[m] - a0) / crb, "o", ms=3, color=c)
    lims = [ax.axis() for ax in axes[:2]]
    bad = ~cert                                           # training not certified: shown as crosses
    axes[0].plot((tree[bad] - a0) / crb, (pinn[bad] - a0) / crb, "x", ms=4, color=COLORS["grey"],
                 label=f"not certified ({int(bad.sum())})")
    axes[1].plot((mle[bad] - a0) / crb, (pinn[bad] - a0) / crb, "x", ms=4, color=COLORS["grey"])
    for ax, l in zip(axes[:2], lims):
        if cert.any():
            ax.axis(l)
    for ax, lab in zip(axes[:2], ("exact tree-level estimate", "hard-constrained MLE")):
        lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lim, lim, color=COLORS["grey"], lw=0.8)
        ax.set_xlabel(f"{lab}:  $(\\hat\\alpha-\\alpha_0)/$CRB")
        ax.set_ylabel("trained PINN:  $(\\hat\\alpha-\\alpha_0)/$CRB")
    axes[0].legend(fontsize=7)
    S = r["summary"]
    x = [s["rho"] for s in S]
    g = lambda s, k: s.get("cert_" + k, np.nan)
    axes[2].errorbar(x, [g(s, "pinn_bias") for s in S], yerr=[g(s, "pinn_bias_err") for s in S], fmt="o",
                     color=COLORS["pinn"], label="certified PINNs: bias")
    axes[2].plot(x, [s["theory_bias"] for s in S], "-", color=COLORS["tree"], label="exact theory: bias")
    axes[2].plot(x, [g(s, "pinn_sd") for s in S], "s", color=COLORS["adam"], label="certified PINNs: std. dev.")
    axes[2].plot(x, [s["theory_sd"] for s in S], "--", color=COLORS["oneloop"], label="exact theory: std. dev.")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("$\\rho$")
    axes[2].set_ylabel("units of CRB")
    axes[2].legend(fontsize=7)
    fig.savefig(out / "pinn_ensemble.png")
    plt.close(fig)
