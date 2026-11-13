"""Helpers shared by the neural experiments."""

from __future__ import annotations

import numpy as np
import torch

from ..exact import (HeatBox, World, argmin_refined, beta_from_rho, cramer_rao, objective_curves,
                     sensor_grid, simulate, weights_ratio_from_rho)
from ..nn import HeatPINN, NetSpec, Weights, lm_fit


def inverse_setup(cfg: dict) -> dict:
    """Problem, sensors, pseudo-data and the exact reference estimates for every data set."""
    problem = HeatBox(dim=1, n_modes=cfg.get("n_modes", 160))
    X, t = sensor_grid(1, cfg["n_space"], cfg["n_time"], problem.T)
    a0, sigma, n = cfg["alpha0"], cfg["sigma"], len(t)
    rng = np.random.default_rng(cfg["seed"])
    Y = simulate(problem, X, t, sigma, a0, World("exact"), cfg["n_datasets"], rng)
    pv = float(problem.kernel_diag(a0, X, t).mean())
    crb = cramer_rao(problem, X, t, sigma, a0)
    return dict(problem=problem, X=X, t=t, Y=Y, pv=pv, crb=crb, n=n, alpha0=a0, sigma=sigma)


def exact_reference(s: dict, rhos, span: float = 16.0, n_grid: int = 321) -> dict:
    """MLE and tree-level estimates (and the ideal loss values) for every data set and rho."""
    a0, crb, sigma, n = s["alpha0"], s["crb"], s["sigma"], s["n"]
    grid = np.linspace(max(a0 - span * crb, 0.05 * a0), a0 + span * crb, n_grid)
    betas = [beta_from_rho(r, sigma, s["pv"]) for r in rhos]
    cur = objective_curves(s["problem"], s["X"], s["t"], s["Y"], sigma, grid, betas)
    mle = argmin_refined(grid, cur.mle)[0]
    tree = np.stack([argmin_refined(grid, cur.tree[b])[0] for b in range(len(rhos))])       # (R, M)
    # ideal value of the PINN loss (w_data = 1):  tree * sigma^2 / N
    loss = np.stack([argmin_refined(grid, cur.tree[b])[1] for b in range(len(rhos))]) * sigma ** 2 / n
    ratios = [weights_ratio_from_rho(r, n, s["problem"].volume, s["pv"]) for r in rhos]
    return dict(mle=mle, tree=tree, ideal_loss=loss, ratios=ratios, grid=grid)


def tree_estimate_at_ratio(s: dict, y: np.ndarray, ratio: float, span: float = 25.0,
                           n_grid: int = 401) -> float:
    """Tree-level estimate for one data set at an arbitrary loss-weight ratio w_res/w_data."""
    a0, crb, sigma, n = s["alpha0"], s["crb"], s["sigma"], s["n"]
    grid = np.linspace(max(a0 - span * crb, 0.05 * a0), a0 + span * crb, n_grid)
    beta = ratio * n / (sigma ** 2 * s["problem"].volume)
    cur = objective_curves(s["problem"], s["X"], s["t"], y, sigma, grid, [beta])
    return float(argmin_refined(grid, cur.tree[0])[0][0])


def net_spec(cfg: dict, in_dim: int, fourier: bool) -> NetSpec:
    return NetSpec(in_dim=in_dim, width=cfg["width"], depth=cfg["depth"],
                   fourier_features=cfg.get("fourier_features", 16) if fourier else 0,
                   fourier_scale=cfg.get("fourier_scale", 3.0), rwf=cfg.get("rwf", False))


def member_layout(n_datasets: int, ratios) -> tuple[np.ndarray, np.ndarray]:
    """Ensemble layout: member e <-> (dataset index, rho index), rho-major."""
    r_idx = np.repeat(np.arange(len(ratios)), n_datasets)
    d_idx = np.tile(np.arange(n_datasets), len(ratios))
    return d_idx, r_idx


def lm_in_groups(pinn: HeatPINN, P0: torch.Tensor, args: tuple, n_iter: int, group_size: int,
                 verbose: bool = True, alpha_warmup: int = 0
                 ) -> tuple[torch.Tensor, np.ndarray, np.ndarray, float]:
    """Run Levenberg-Marquardt on sub-groups of the ensemble to bound memory.

    ``alpha_warmup`` iterations are run with the diffusivity frozen at its initial
    value.  A randomly initialised network has u_t ~ 0, so the residual is
    proportional to alpha and the optimiser would otherwise drive alpha towards
    zero before the field has formed (a spurious attractor).

    Returns parameters, loss history (iters+1, E), alpha trajectory (iters+1, E) and wall time.
    """
    E = P0.shape[0]
    frozen = None
    if pinn.learn_alpha and alpha_warmup > 0:
        frozen = torch.zeros(P0.shape[1], dtype=torch.bool, device=P0.device)
        frozen[-1] = True
    P_out, loss_h, alpha_h, wall = [], [], [], 0.0
    for s in range(0, E, group_size):
        sl = slice(s, min(s + group_size, E))
        traj = [pinn.alpha_of(P0[sl]).cpu().numpy().copy()] if pinn.learn_alpha else []
        cb = (lambda it, p, l: traj.append(pinn.alpha_of(p).cpu().numpy().copy())) if pinn.learn_alpha else None
        if verbose:
            print(f"  LM group {s // group_size + 1}/{(E + group_size - 1) // group_size} "
                  f"({sl.stop - sl.start} networks, {pinn.n_params} parameters each)", flush=True)
        P, info = lm_fit(pinn.system, pinn.loss, P0[sl], tuple(a[sl] for a in args), n_iter=n_iter,
                         callback=cb, verbose=verbose, frozen=frozen, freeze_iters=alpha_warmup)
        H = info["loss"].numpy()
        pad = n_iter + 1 - H.shape[0]
        if pad > 0:                                           # early convergence: hold last value
            H = np.concatenate([H, np.repeat(H[-1:], pad, axis=0)])
            if traj:
                traj += [traj[-1]] * pad
        P_out.append(P)
        loss_h.append(H)
        if traj:
            alpha_h.append(np.stack(traj))
        wall += info["time"]
    alpha = np.concatenate(alpha_h, axis=1) if alpha_h else np.zeros((0, E))
    return torch.cat(P_out), np.concatenate(loss_h, axis=1), alpha, wall


@torch.no_grad()
def dense_residual(pinn: HeatPINN, P: torch.Tensor, args: tuple, n_points: int = 16384,
                   seed: int = 777) -> np.ndarray:
    """Mean-square PDE residual on fresh points (tests for collocation over-fitting)."""
    old_n, old_Zc = pinn.n_colloc, pinn.Zc
    state = {k: getattr(pinn, k) for k in ("Zi", "u0_i", "Zb") if hasattr(pinn, k)}
    pinn.n_colloc = n_points
    pinn.resample(seed)
    out = []
    for s in range(0, P.shape[0], 8):
        sl = slice(s, s + 8)
        L = pinn.term_losses(P[sl], args[0][sl], args[1][sl])
        out.append(L[pinn.term_names.index("res")].cpu().numpy())
    pinn.n_colloc, pinn.Zc = old_n, old_Zc
    for k, v in state.items():
        setattr(pinn, k, v)
    return np.concatenate(out)
