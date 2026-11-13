"""Closed-form estimators of the diffusivity defined by PINN-type objectives.

Notation (see docs/THEORY.md)
-----------------------------
data            y_i = u(z_i) + eps_i,   eps ~ N(0, sigma^2),  z_i = (x_i, t_i)
PINN objective  S[u, alpha] = sum_i (u(z_i) - y_i)^2 / sigma^2
                              + beta * int int (u_t - alpha Lap u)^2 dx dt
                (initial and boundary conditions imposed exactly)

For fixed alpha the minimisation over the *field* u can be done in closed form:
with r = y - m_alpha (m_alpha = exact PDE solution at the sensors) and
C_alpha the stochastic-heat-equation kernel,

    tree(alpha)     = min_u S              = r^T (sigma^2 I + C_alpha/beta)^-1 r
    oneloop(alpha)  = -2 log p(y | alpha)  = tree + log det(I + C_alpha/(beta sigma^2))
    mle(alpha)      = lim_{beta->inf} tree = |r|^2 / sigma^2
    profiled(alpha) = data misfit of the tree-level field
                    = sigma^2 r^T (sigma^2 I + C_alpha/beta)^-2 r

* ``tree``     is what a (perfectly trained, infinitely wide) inverse PINN minimises.
* ``oneloop``  adds the Gaussian fluctuation determinant, i.e. it is the exact
               marginal likelihood of the "soft physics" statistical model.
* ``mle``      is the classical hard-constrained least-squares fit.
* ``profiled`` is the outer criterion of generalised profiling / bilevel
               schemes evaluated on the soft-constrained inner solution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spectral import HeatBox


# ------------------------------------------------------------------ conversions
def beta_from_weights(w_data: float, w_res: float, n_data: int, sigma: float, volume: float) -> float:
    """Physics precision beta implied by PINN loss weights.

    PINN loss:  w_data * mean_i (u - y)^2 + w_res * mean_colloc r^2, where the
    collocation mean approximates (1/volume) * int r^2.  Multiplying by
    n_data / (w_data sigma^2) gives S with  beta = (w_res/w_data) n_data / (sigma^2 volume).
    """
    return (w_res / w_data) * n_data / (sigma ** 2 * volume)


def rho_from_beta(beta: float, sigma: float, prior_var_unit: float) -> float:
    """rho = (prior std of the model-error field at the sensors) / (noise std)."""
    return float(np.sqrt(prior_var_unit / beta) / sigma)


def beta_from_rho(rho: float, sigma: float, prior_var_unit: float) -> float:
    return float(prior_var_unit / (rho * sigma) ** 2)


def weights_ratio_from_rho(rho: float, n_data: int, volume: float, prior_var_unit: float) -> float:
    """w_res / w_data that realises a given rho (independent of sigma)."""
    return prior_var_unit * volume / (rho ** 2 * n_data)


def cramer_rao(problem: HeatBox, X, t, sigma: float, alpha: float,
               beta: float | None = None, beta_known: bool = True, rel_step: float = 1e-4) -> float:
    """Cramer-Rao bound (standard deviation) for alpha.

    beta=None: deterministic physics, Fisher = |dm/dalpha|^2 / sigma^2.
    Otherwise the data are N(m_alpha, sigma^2 I + C_alpha / beta); if
    ``beta_known`` is False the bound accounts for jointly estimating log(beta).
    """
    dm = problem.dmean_dalpha(alpha, X, t)
    if beta is None:
        return float(sigma / np.sqrt(dm @ dm))
    n = len(t)
    h = rel_step * alpha
    S = sigma ** 2 * np.eye(n) + problem.kernel(alpha, X, t) / beta
    dS_a = (problem.kernel(alpha + h, X, t) - problem.kernel(alpha - h, X, t)) / (2 * h * beta)
    Si = np.linalg.inv(S)
    A = Si @ dS_a
    F_aa = dm @ Si @ dm + 0.5 * np.trace(A @ A)
    if beta_known:
        return float(1.0 / np.sqrt(F_aa))
    dS_b = -(S - sigma ** 2 * np.eye(n))          # d Sigma / d log(beta)
    B = Si @ dS_b
    F_ab = 0.5 * np.trace(A @ B)
    F_bb = 0.5 * np.trace(B @ B)
    return float(np.sqrt(F_bb / (F_aa * F_bb - F_ab ** 2)))


# ------------------------------------------------------------- objective curves
@dataclass
class ObjectiveCurves:
    """Objective functions on an alpha grid for M data sets and B values of beta.

    Shapes: mle (G, M); tree, oneloop, profiled (B, G, M).
    All are in units of -2 log-likelihood, so a unit increase corresponds to
    "one sigma" for a correctly specified Gaussian likelihood.
    """

    alpha: np.ndarray
    beta: np.ndarray
    mle: np.ndarray
    tree: np.ndarray
    oneloop: np.ndarray
    profiled: np.ndarray

    def learned(self) -> np.ndarray:
        """One-loop objective profiled over beta (including beta = infinity). (G, M)"""
        return np.minimum(self.oneloop.min(axis=0), self.mle)

    def learned_beta_index(self) -> np.ndarray:
        """Index of the selected beta per data set at the optimum (-1 means beta = inf)."""
        J = np.concatenate([self.oneloop, self.mle[None]], axis=0)        # (B+1, G, M)
        flat = J.reshape(-1, J.shape[-1]).argmin(axis=0)
        b = flat // J.shape[1]
        return np.where(b == len(self.beta), -1, b)


def objective_curves(problem: HeatBox, X, t, Y, sigma: float, alpha_grid, betas) -> ObjectiveCurves:
    """Evaluate all objectives for data sets Y (M, N) (or a single (N,) vector)."""
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    alpha_grid = np.asarray(alpha_grid, dtype=float)
    betas = np.atleast_1d(np.asarray(betas, dtype=float))
    G, B, M = len(alpha_grid), len(betas), Y.shape[0]
    mle = np.empty((G, M))
    tree = np.empty((B, G, M))
    oneloop = np.empty((B, G, M))
    profiled = np.empty((B, G, M))
    s2 = sigma ** 2
    for g, a in enumerate(alpha_grid):
        R = Y - problem.mean(a, X, t)[None, :]
        lam, Q = np.linalg.eigh(problem.kernel(a, X, t))
        lam = np.clip(lam, 0.0, None)
        Z2 = (R @ Q) ** 2                                   # (M, N)
        mle[g] = (R ** 2).sum(1) / s2
        for b, be in enumerate(betas):
            d = s2 + lam / be
            tree[b, g] = (Z2 / d).sum(1)
            oneloop[b, g] = tree[b, g] + np.log1p(lam / (be * s2)).sum()
            profiled[b, g] = s2 * (Z2 / d ** 2).sum(1)
    return ObjectiveCurves(alpha_grid, betas, mle, tree, oneloop, profiled)


def tree_level_field(problem: HeatBox, X, t, y, sigma: float, alpha: float, beta: float,
                     Xq, tq) -> np.ndarray:
    """The minimiser over u of the PINN objective at fixed alpha, evaluated at (Xq, tq).

    u_hat = m_alpha + C(q, data) (sigma^2 beta I + C(data, data))^-1 (y - m_alpha)
    This is the field an ideal PINN would return.
    """
    r = np.asarray(y, dtype=float) - problem.mean(alpha, X, t)
    K = sigma ** 2 * beta * np.eye(len(t)) + problem.kernel(alpha, X, t)
    w = np.linalg.solve(K, r)
    return problem.mean(alpha, Xq, tq) + problem.kernel(alpha, Xq, tq, X, t) @ w


# ----------------------------------------------------------- estimates/intervals
def argmin_refined(grid: np.ndarray, J: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Minimise J (G, M) over the grid with a parabolic refinement.

    Returns (estimate, objective value at the minimum, at_edge flag).
    """
    grid = np.asarray(grid, dtype=float)
    am = J.argmin(axis=0)
    edge = (am == 0) | (am == len(grid) - 1)
    i = np.clip(am, 1, len(grid) - 2)
    j = np.arange(J.shape[1])
    f0, f1, f2 = J[i - 1, j], J[i, j], J[i + 1, j]
    den = f0 - 2.0 * f1 + f2
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(den > 0, 0.5 * (f0 - f2) / den, 0.0)
    d = np.clip(d, -1.0, 1.0)
    h = grid[1] - grid[0]
    est = np.where(edge, grid[am], grid[i] + d * h)
    fmin = np.where(edge, J[am, j], f1 - 0.25 * (f0 - f2) * d)
    return est, fmin, edge


def delta_interval(grid: np.ndarray, J: np.ndarray, delta: float = 1.0
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interval {alpha : J(alpha) <= min J + delta} with linear interpolation.

    Returns (lo, hi, open) where ``open`` flags intervals truncated by the grid.
    """
    grid = np.asarray(grid, dtype=float)
    G, M = J.shape
    _, fmin, _ = argmin_refined(grid, J)
    am = J.argmin(axis=0)
    D = J - (fmin + delta)[None, :]                      # <= 0 inside
    lo = np.empty(M)
    hi = np.empty(M)
    open_ = np.zeros(M, dtype=bool)
    for m in range(M):
        d = D[:, m]
        k = am[m]
        # walk left
        i = k
        while i > 0 and d[i - 1] <= 0:
            i -= 1
        if i == 0:
            lo[m] = grid[0]
            open_[m] = True
        else:
            lo[m] = grid[i - 1] + (grid[i] - grid[i - 1]) * d[i - 1] / (d[i - 1] - d[i])
        i = k
        while i < G - 1 and d[i + 1] <= 0:
            i += 1
        if i == G - 1:
            hi[m] = grid[-1]
            open_[m] = True
        else:
            hi[m] = grid[i] + (grid[i + 1] - grid[i]) * (-d[i]) / (d[i + 1] - d[i])
    return lo, hi, open_
