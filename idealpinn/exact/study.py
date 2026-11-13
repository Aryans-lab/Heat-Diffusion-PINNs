"""Pseudo-experiment studies: frequentist properties of the closed-form estimators."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .spectral import HeatBox
from .estimators import (ObjectiveCurves, argmin_refined, beta_from_rho, cramer_rao,
                         delta_interval, objective_curves)

NORMAL_68 = 0.6826894921370859


@dataclass
class World:
    """How the pseudo-data are generated (the estimators always assume the pure heat equation).

    kind = "exact"       u_t = alpha u_xx                         (model is right)
           "stochastic"  u_t = alpha u_xx + xi, noise amplitude set by rho_true
           "loss"        u_t = alpha u_xx - loss_rate * u         (unmodelled cooling)
           "source"      u_t = alpha u_xx + source                (unmodelled heating)
    """

    kind: str = "exact"
    rho_true: float = 0.0
    loss_rate: float = 0.0
    source: float = 0.0


def simulate(problem: HeatBox, X, t, sigma: float, alpha0: float, world: World,
             n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n pseudo-experiments (n, N)."""
    kw = {}
    if world.kind == "loss":
        kw["loss_rate"] = world.loss_rate
    elif world.kind == "source":
        kw["source"] = world.source
    elif world.kind not in ("exact", "stochastic"):
        raise ValueError(f"unknown world {world.kind}")
    Y = problem.mean(alpha0, X, t, **kw)[None, :] + sigma * rng.standard_normal((n, len(t)))
    if world.kind == "stochastic":
        pv = float(problem.kernel_diag(alpha0, X, t).mean())
        beta_true = beta_from_rho(world.rho_true, sigma, pv)
        Y = Y + problem.sample_stochastic(alpha0, X, t, beta_true, n, rng)
    return Y


def summarise(grid: np.ndarray, J: np.ndarray, alpha0: float, unit: float) -> dict:
    """Bias / spread / coverage of the estimator argmin J, in units of ``unit``."""
    est, _, edge = argmin_refined(grid, J)
    lo, hi, open_ = delta_interval(grid, J, 1.0)
    half = 0.5 * (hi - lo)
    with np.errstate(divide="ignore", invalid="ignore"):
        pull = (est - alpha0) / half
    M = len(est)
    bias = est.mean() - alpha0
    sd = est.std(ddof=1)
    cover = np.mean((lo <= alpha0) & (alpha0 <= hi))
    return {
        "bias": float(bias / unit),
        "bias_err": float(sd / np.sqrt(M) / unit),
        "sd": float(sd / unit),
        "sd_err": float(sd / np.sqrt(2 * (M - 1)) / unit),
        "rmse": float(np.sqrt(np.mean((est - alpha0) ** 2)) / unit),
        "coverage": float(cover),
        "coverage_err": float(np.sqrt(max(cover * (1 - cover), 1e-12) / M)),
        "halfwidth": float(np.median(half) / unit),
        "pull_mean": float(np.nanmean(pull)),
        "pull_sd": float(np.nanstd(pull)),
        "edge_frac": float(np.mean(edge | open_)),
        "n": int(M),
    }


def run_study(problem: HeatBox, X, t, sigma: float, alpha0: float, world: World,
              rhos, n_experiments: int = 2000, seed: int = 0, span: float = 14.0,
              n_grid: int = 281, chunk: int = 2000, return_estimates: bool = False) -> dict:
    """Frequentist study of all estimators for one measurement design.

    ``rhos`` are the assumed model-error-to-noise ratios (equivalently PINN loss
    weights, see :func:`idealpinn.exact.weights_ratio_from_rho`).  Results are
    reported in units of the Cramer-Rao bound of the *exact* world, so "sd = 1"
    means statistically efficient.
    """
    rng = np.random.default_rng(seed)
    rhos = np.atleast_1d(np.asarray(rhos, dtype=float))
    pv = float(problem.kernel_diag(alpha0, X, t).mean())
    betas = np.array([beta_from_rho(r, sigma, pv) for r in rhos])
    crb = cramer_rao(problem, X, t, sigma, alpha0)
    width = crb
    crb_world = crb
    if world.kind == "stochastic":
        bt = beta_from_rho(world.rho_true, sigma, pv)
        crb_world = cramer_rao(problem, X, t, sigma, alpha0, beta=bt, beta_known=False)
        width = max(crb, crb_world)
    lo = max(alpha0 - span * width, 0.05 * alpha0)
    grid = np.linspace(lo, alpha0 + span * width, n_grid)

    acc: dict[str, list] = {}
    for s in range(0, n_experiments, chunk):
        Y = simulate(problem, X, t, sigma, alpha0, world, min(chunk, n_experiments - s), rng)
        cur = objective_curves(problem, X, t, Y, sigma, grid, betas)
        acc.setdefault("mle", []).append(cur.mle)
        acc.setdefault("oneloop_learned", []).append(cur.learned())
        for b, r in enumerate(rhos):
            acc.setdefault(f"tree@{r:g}", []).append(cur.tree[b])
            acc.setdefault(f"oneloop@{r:g}", []).append(cur.oneloop[b])
            acc.setdefault(f"profiled@{r:g}", []).append(cur.profiled[b])
    out = {
        "meta": {
            "world": asdict(world), "sigma": sigma, "alpha0": alpha0, "n_data": int(len(t)),
            "dim": problem.dim, "n_modes": problem.n_modes, "crb": crb, "crb_world": crb_world,
            "prior_var_unit": pv, "rhos": rhos.tolist(), "betas": betas.tolist(),
            "grid": [float(grid[0]), float(grid[-1]), int(n_grid)], "seed": seed,
            "weights_ratio": [pv * problem.volume / (r ** 2 * len(t)) for r in rhos],
        },
        "estimators": {},
    }
    for name, parts in acc.items():
        J = np.concatenate(parts, axis=1)
        out["estimators"][name] = summarise(grid, J, alpha0, crb)
        if return_estimates:
            out["estimators"][name]["estimates"] = argmin_refined(grid, J)[0].tolist()
    return out
