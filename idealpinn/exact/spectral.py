"""Exact spectral description of the heat equation on the unit box.

Problem
-------
    u_t = alpha * Laplacian(u)      on  [0, 1]^d x [0, T]
    u = 0                           on the boundary
    u(x, 0) = u0(x)                 (a band-limited "Gaussian bump")

With the Dirichlet eigenfunctions

    phi_k(x) = prod_i sqrt(2) sin(pi k_i x_i),     mu_k = pi^2 |k|^2,

the solution is  u = sum_k c_k exp(-alpha mu_k t) phi_k(x).

The class below also provides the covariance kernel of the *stochastic* heat
equation  u_t - alpha Lap(u) = xi  with space-time white noise of unit
intensity and u(., 0) = 0:

    C_alpha(z, z') = sum_k phi_k(x) phi_k(x')
                     * [exp(-a_k |t - t'|) - exp(-a_k (t + t'))] / (2 a_k),
    a_k = alpha mu_k.

This kernel is the central object of the theory: the L2 PDE-residual penalty
used by a PINN is exactly the negative log-density of this Gaussian field
(see docs/THEORY.md, section 2).

A unit box is used without loss of generality: a box of side L is recovered by
the substitution alpha -> alpha / L^2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np

SQRT2 = np.sqrt(2.0)


def gaussian_sine_coefficients(n_modes: int, width: float, centre: float = 0.5,
                               n_quad: int = 20001) -> np.ndarray:
    """Sine coefficients c_k = int_0^1 g(x) sqrt(2) sin(k pi x) dx of a Gaussian."""
    x = np.linspace(0.0, 1.0, n_quad)
    g = np.exp(-((x - centre) ** 2) / (2.0 * width ** 2))
    k = np.arange(1, n_modes + 1)
    basis = SQRT2 * np.sin(np.pi * np.outer(x, k))
    # composite Simpson (n_quad is odd)
    w = np.ones(n_quad)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    w *= (x[1] - x[0]) / 3.0
    return (w * g) @ basis


@dataclass
class HeatBox:
    """Heat equation on [0,1]^dim with zero Dirichlet walls.

    Parameters
    ----------
    dim : spatial dimension (1, 2 or 3).
    T : final time.
    n_modes : spectral cut-off of the *kernel* (modes per axis for
        ``cutoff="box"``; radius in index space for ``cutoff="ball"``).
    ic_modes : number of sine modes (per axis) kept in the initial condition.
        The initial condition *is* this finite sum, so it vanishes exactly on
        the walls and the exact solution is known to machine precision.
    ic_width : width of the Gaussian that is projected on the sine modes.
    ic_amp : overall amplitude of the initial condition.
    cutoff : "box" (k_i <= n_modes) or "ball" (|k| <= n_modes).
    """

    dim: int = 1
    T: float = 1.0
    n_modes: int = 128
    ic_modes: int = 16
    ic_width: float = 0.1
    ic_amp: float = 1.0
    cutoff: str = "box"

    idx: np.ndarray = field(init=False, repr=False)       # kernel modes (K, d)
    mu: np.ndarray = field(init=False, repr=False)        # eigenvalues  (K,)
    ic_idx: np.ndarray = field(init=False, repr=False)    # IC modes     (K0, d)
    ic_mu: np.ndarray = field(init=False, repr=False)
    ic_coef: np.ndarray = field(init=False, repr=False)   # IC coefficients (K0,)
    ic_coef_1d: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.dim not in (1, 2, 3):
            raise ValueError("dim must be 1, 2 or 3")
        rng1 = np.arange(1, self.n_modes + 1)
        idx = np.array(list(product(rng1, repeat=self.dim)), dtype=np.int64)
        if self.cutoff == "ball":
            idx = idx[(idx ** 2).sum(1) <= self.n_modes ** 2]
        elif self.cutoff != "box":
            raise ValueError("cutoff must be 'box' or 'ball'")
        self.idx = idx
        self.mu = np.pi ** 2 * (idx ** 2).sum(1).astype(float)

        c1 = gaussian_sine_coefficients(self.ic_modes, self.ic_width)
        # normalise so that the 1D factor has unit maximum -> u0 peaks at ic_amp
        xs = np.linspace(0, 1, 2001)
        f = (SQRT2 * np.sin(np.pi * np.outer(xs, np.arange(1, self.ic_modes + 1)))) @ c1
        c1 = c1 / f.max()
        self.ic_coef_1d = c1 * self.ic_amp ** (1.0 / self.dim)
        r0 = np.arange(1, self.ic_modes + 1)
        ic_idx = np.array(list(product(r0, repeat=self.dim)), dtype=np.int64)
        self.ic_idx = ic_idx
        self.ic_mu = np.pi ** 2 * (ic_idx ** 2).sum(1).astype(float)
        self.ic_coef = np.prod(self.ic_coef_1d[ic_idx - 1], axis=1)

    # ------------------------------------------------------------------ basis
    @staticmethod
    def _phi(X: np.ndarray, idx: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        out = np.ones((X.shape[0], idx.shape[0]))
        for i in range(idx.shape[1]):
            out *= SQRT2 * np.sin(np.pi * np.outer(X[:, i], idx[:, i]))
        return out

    def phi(self, X: np.ndarray) -> np.ndarray:
        """Kernel-mode eigenfunctions at points X (N, d) -> (N, K)."""
        return self._phi(X, self.idx)

    @property
    def volume(self) -> float:
        """Space-time volume |Omega| * T of the domain."""
        return float(self.T)

    # ------------------------------------------------------- deterministic part
    def initial_condition(self, X: np.ndarray) -> np.ndarray:
        return self._phi(X, self.ic_idx) @ self.ic_coef

    def mean(self, alpha: float, X: np.ndarray, t: np.ndarray, loss_rate: float = 0.0,
             source: float = 0.0) -> np.ndarray:
        """Exact solution m_alpha at space-time points (X[i], t[i]).

        ``loss_rate`` h adds an (unmodelled) Newton-cooling term -h u to the
        dynamics, ``source`` s adds a constant uniform heating +s; both are only
        used to generate mis-specified data, never by the estimators.
        """
        X = np.atleast_2d(X)
        t = np.asarray(t, dtype=float)
        P = self._phi(X, self.ic_idx)
        a = alpha * self.ic_mu + loss_rate
        out = (P * np.exp(-np.outer(t, a))) @ self.ic_coef
        if source != 0.0:
            # response to a constant unit source: sum_k <1, phi_k> (1 - e^{-a t}) / a
            Pk = self.phi(X)
            ak = alpha * self.mu + loss_rate
            proj = np.ones(self.idx.shape[0])
            for i in range(self.dim):
                k = self.idx[:, i]
                proj *= SQRT2 * (1.0 - np.cos(np.pi * k)) / (np.pi * k)
            out = out + source * ((Pk * (-np.expm1(-np.outer(t, ak)) / ak)) @ proj)
        return out

    def dmean_dalpha(self, alpha: float, X: np.ndarray, t: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        t = np.asarray(t, dtype=float)
        P = self._phi(X, self.ic_idx)
        e = np.exp(-alpha * np.outer(t, self.ic_mu))
        return (P * e * (-np.outer(t, self.ic_mu))) @ self.ic_coef

    def solution_on_grid(self, alpha: float, n: int, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Exact field on a uniform n^d grid at time t. Returns (points, values)."""
        g = np.linspace(0.0, 1.0, n)
        X = np.array(list(product(g, repeat=self.dim)))
        return X, self.mean(alpha, X, np.full(X.shape[0], t))

    # ----------------------------------------------------------- stochastic part
    def _time_factor(self, alpha: float, t: np.ndarray, t2: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """g_k(t_i, t'_j) for a block of modes -> (N, N2, Kblock)."""
        a = alpha * mu
        dt = np.abs(t[:, None] - t2[None, :])[..., None]
        st = (t[:, None] + t2[None, :])[..., None]
        return (np.exp(-a * dt) - np.exp(-a * st)) / (2.0 * a)

    def kernel(self, alpha: float, X: np.ndarray, t: np.ndarray,
               X2: np.ndarray | None = None, t2: np.ndarray | None = None,
               block: int = 256) -> np.ndarray:
        """Covariance C_alpha between (X, t) and (X2, t2) for unit noise intensity."""
        X = np.atleast_2d(X)
        t = np.asarray(t, dtype=float)
        same = X2 is None
        X2 = X if same else np.atleast_2d(X2)
        t2 = t if t2 is None else np.asarray(t2, dtype=float)
        C = np.zeros((X.shape[0], X2.shape[0]))
        for s in range(0, self.idx.shape[0], block):
            idx = self.idx[s:s + block]
            P = self._phi(X, idx)
            P2 = P if same else self._phi(X2, idx)
            G = self._time_factor(alpha, t, t2, self.mu[s:s + block])
            C += np.einsum("ik,jk,ijk->ij", P, P2, G, optimize=True)
        if same:
            C = 0.5 * (C + C.T)
        return C

    def kernel_diag(self, alpha: float, X: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Prior variance C_alpha(z, z) of the unit-intensity stochastic field."""
        X = np.atleast_2d(X)
        t = np.asarray(t, dtype=float)
        out = np.zeros(X.shape[0])
        block = 4096
        for s in range(0, self.idx.shape[0], block):
            P = self._phi(X, self.idx[s:s + block])
            a = alpha * self.mu[s:s + block]
            out += ((P ** 2) * (-np.expm1(-2.0 * np.outer(t, a))) / (2.0 * a)).sum(1)
        return out

    def sample_stochastic(self, alpha: float, X: np.ndarray, t: np.ndarray, beta: float,
                          n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw n realisations of the model-error field (intensity 1/beta) at the sensors."""
        C = self.kernel(alpha, X, t) / beta
        lam, Q = np.linalg.eigh(C)
        A = Q * np.sqrt(np.clip(lam, 0.0, None))
        return rng.standard_normal((n, C.shape[0])) @ A.T


# ---------------------------------------------------------------------- sensors
def sensor_grid(dim: int, n_space: int, n_time: int, T: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """n_space^dim interior sensors, each read out at n_time equispaced times in (0, T].

    Returns X (N, dim), t (N,) with N = n_space^dim * n_time; the ordering is
    "sensor-major" (all times of sensor 0, then sensor 1, ...).
    """
    g = np.linspace(0.0, 1.0, n_space + 2)[1:-1]
    pts = np.array(list(product(g, repeat=dim)))
    ts = np.linspace(0.0, T, n_time + 1)[1:]
    X = np.repeat(pts, n_time, axis=0)
    t = np.tile(ts, pts.shape[0])
    return X, t


def sensor_scatter(dim: int, n: int, T: float = 1.0, seed: int = 0,
                   margin: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """n scattered space-time measurement points (scrambled Sobol)."""
    from scipy.stats import qmc

    s = qmc.Sobol(dim + 1, scramble=True, seed=seed).random(n)
    X = margin + (1.0 - 2.0 * margin) * s[:, :dim]
    t = T * (0.02 + 0.98 * s[:, dim])
    return X, t
