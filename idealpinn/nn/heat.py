"""PINN formulation of the heat equation on the unit box (forward and inverse)."""

from __future__ import annotations

from dataclasses import dataclass

import math
import numpy as np
import torch
from scipy.stats import qmc
from torch.func import grad, vmap

from ..exact.spectral import HeatBox
from .model import NetSpec, forward_jet, init_params

SQRT2 = math.sqrt(2.0)


@dataclass
class Weights:
    """Loss weights:  loss = sum_k w_k * mean(term_k^2)."""

    data: float = 1.0
    res: float = 1.0
    ic: float = 1.0
    bc: float = 1.0


class HeatPINN:
    """Residuals of a (batched) PINN for u_t = alpha Lap(u) on [0,1]^d x [0,T].

    Parameters
    ----------
    constraint :
        "ic_bc"  u = u0(x) + t * B(x) * N(x,t)   initial + boundary conditions exact
        "bc"     u = B(x) * N(x,t)               boundary exact, initial condition free
        "none"   u = N(x,t)                      everything soft (the classical PINN)
        with B(x) = prod_i 4 x_i (1 - x_i).
    sensors : optional (X, t) measurement locations; enables the data term.
    soft_ic : add an initial-condition loss term when the constraint does not
        impose it exactly (set False for inverse problems with unknown initial state).
    learn_alpha : if True the last entry of each parameter vector is
        eta = log(alpha / alpha_init) and alpha is inferred (inverse problem).
    """

    def __init__(self, problem: HeatBox, spec: NetSpec, constraint: str = "ic_bc",
                 n_colloc: int = 2048, n_ic: int = 256, n_bc: int = 256,
                 sensors: tuple[np.ndarray, np.ndarray] | None = None,
                 learn_alpha: bool = False, alpha_init: float = 0.05,
                 soft_ic: bool = True, device: str | None = None, dtype=torch.float64, seed: int = 0):
        if constraint not in ("ic_bc", "bc", "none"):
            raise ValueError(constraint)
        assert spec.in_dim == problem.dim + 1
        self.problem, self.spec, self.constraint = problem, spec, constraint
        self.dim = problem.dim
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype
        self.learn_alpha = learn_alpha
        self.alpha_init = alpha_init
        self.n_colloc, self.n_ic, self.n_bc = n_colloc, n_ic, n_bc
        self.point_chunk = 1024                # points per Jacobian block in Levenberg-Marquardt
        self._c1 = self._t(problem.ic_coef_1d)
        self._k1 = self._t(np.arange(1, problem.ic_modes + 1) * np.pi)
        self.Zd = None
        if sensors is not None:
            X, t = sensors
            self.Zd = self._t(np.concatenate([np.atleast_2d(X), np.asarray(t)[:, None]], axis=1))
        self.term_names = (["data"] if self.Zd is not None else []) + ["res"]
        if constraint in ("none", "bc") and soft_ic and n_ic > 0:
            self.term_names.append("ic")
        if constraint == "none" and n_bc > 0:
            self.term_names.append("bc")
        self.resample(seed)

    # ---------------------------------------------------------------- helpers
    def _t(self, a) -> torch.Tensor:
        return torch.as_tensor(np.asarray(a), dtype=self.dtype, device=self.device)

    def resample(self, seed: int) -> None:
        """Draw new collocation / initial / boundary points (scrambled Sobol)."""
        d, T = self.dim, self.problem.T
        s = qmc.Sobol(d + 1, scramble=True, seed=seed).random(self.n_colloc)
        s[:, d] *= T
        self.Zc = self._t(s)
        if "ic" in self.term_names:
            s = qmc.Sobol(d, scramble=True, seed=seed + 1).random(self.n_ic)
            self.Zi = self._t(np.concatenate([s, np.zeros((self.n_ic, 1))], axis=1))
            self.u0_i = self._t(self.problem.initial_condition(s))
        if "bc" in self.term_names:
            rng = np.random.default_rng(seed + 2)
            s = rng.random((self.n_bc, d + 1))
            s[:, d] *= T
            face = rng.integers(0, d, self.n_bc)
            s[np.arange(self.n_bc), face] = rng.integers(0, 2, self.n_bc).astype(float)
            self.Zb = self._t(s)

    @property
    def n_params(self) -> int:
        return self.spec.n_params + (1 if self.learn_alpha else 0)

    def init(self, n_ens: int, seed: int = 0) -> torch.Tensor:
        p = init_params(self.spec, n_ens, seed, self.dtype, self.device)
        if self.learn_alpha:
            p = torch.cat([p, torch.zeros(n_ens, 1, dtype=self.dtype, device=self.device)], dim=1)
        return p

    def alpha_of(self, P: torch.Tensor) -> torch.Tensor:
        """Inferred diffusivity of every member (E,)."""
        return self.alpha_init * torch.exp(P[..., -1])

    # ------------------------------------------------------- initial condition
    def _u0(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """u0(x) and its Laplacian for points X (N, d)."""
        s = SQRT2 * torch.sin(X[..., None] * self._k1)             # (N, d, K0)
        f = s @ self._c1                                           # (N, d)
        f2 = -(s * self._k1 ** 2) @ self._c1
        u0 = torch.prod(f, dim=1)
        lap = torch.zeros_like(u0)
        for i in range(self.dim):
            others = torch.prod(torch.cat([f[:, :i], f[:, i + 1:]], dim=1), dim=1) if self.dim > 1 else 1.0
            lap = lap + f2[:, i] * others
        return u0, lap

    # ------------------------------------------------------------------ fields
    def fields(self, p_net: torch.Tensor, Z: torch.Tensor
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """u, u_t and Laplacian(u) of one network at points Z (N, d+1)."""
        d = self.dim
        n, dn, d2n = forward_jet(self.spec, p_net, Z)
        n_t = dn[d]
        if self.constraint == "none":
            return n, n_t, d2n.sum(0)
        X, t = Z[:, :d], Z[:, d]
        q = 4.0 * X * (1.0 - X)                                    # (N, d)
        dq = 4.0 * (1.0 - 2.0 * X)
        Bfac = torch.prod(q, dim=1)
        lap_BN = torch.zeros_like(n)
        for i in range(d):
            others = torch.prod(torch.cat([q[:, :i], q[:, i + 1:]], dim=1), dim=1) if d > 1 else 1.0
            B_i = dq[:, i] * others
            B_ii = -8.0 * others
            lap_BN = lap_BN + B_ii * n + 2.0 * B_i * dn[i] + Bfac * d2n[i]
        if self.constraint == "bc":
            return Bfac * n, Bfac * n_t, lap_BN
        u0, lap_u0 = self._u0(X)
        return u0 + t * Bfac * n, Bfac * n + t * Bfac * n_t, lap_u0 + t * lap_BN

    # ------------------------------------------------------------------- terms
    def terms(self, p: torch.Tensor, y: torch.Tensor, alpha: torch.Tensor) -> list[torch.Tensor]:
        """Unweighted residual vectors of one member, ordered as ``self.term_names``."""
        if self.learn_alpha:
            p_net, alpha = p[:-1], self.alpha_init * torch.exp(p[-1])
        else:
            p_net = p
        out = []
        if self.Zd is not None:
            out.append(self.fields(p_net, self.Zd)[0] - y)
        _, u_t, lap = self.fields(p_net, self.Zc)
        out.append(u_t - alpha * lap)
        if "ic" in self.term_names:
            out.append(self.fields(p_net, self.Zi)[0] - self.u0_i)
        if "bc" in self.term_names:
            out.append(self.fields(p_net, self.Zb)[0])
        return out

    # ------------------------------------------- Gauss-Newton building blocks
    def _point_sets(self, y: torch.Tensor) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
        """(name, points, targets) of every loss term, ordered as ``self.term_names``."""
        sets = []
        for name in self.term_names:
            if name == "data":
                sets.append((name, self.Zd, y))
            elif name == "res":
                sets.append((name, self.Zc, torch.zeros_like(self.Zc[:, 0])))
            elif name == "ic":
                sets.append((name, self.Zi, self.u0_i))
            else:
                sets.append((name, self.Zb, torch.zeros_like(self.Zb[:, 0])))
        return sets

    def _point_residual(self, name: str):
        """Scalar residual of a single point, as a function of the parameters."""
        def f(p, z, target, alpha):
            if self.learn_alpha:
                p_net, alpha = p[:-1], self.alpha_init * torch.exp(p[-1])
            else:
                p_net = p
            u, u_t, lap = self.fields(p_net, z[None])
            val = (u_t - alpha * lap)[0] if name == "res" else (u[0] - target)
            return val, val
        return f

    def system(self, p: torch.Tensor, y: torch.Tensor, alpha: torch.Tensor,
               sqrtw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """loss = |r|^2, A = J^T J, g = J^T r for one member (exact per-point Jacobians)."""
        P = p.shape[0]
        A = torch.zeros(P, P, dtype=p.dtype, device=p.device)
        g = torch.zeros(P, dtype=p.dtype, device=p.device)
        loss = torch.zeros((), dtype=p.dtype, device=p.device)
        for k, (name, Z, target) in enumerate(self._point_sets(y)):
            fn = vmap(grad(self._point_residual(name), has_aux=True), in_dims=(None, 0, 0, None))
            scale = sqrtw[k] ** 2 / Z.shape[0]
            for s in range(0, Z.shape[0], self.point_chunk):          # bounds peak memory
                J, r = fn(p, Z[s:s + self.point_chunk], target[s:s + self.point_chunk], alpha)
                A = A + scale * (J.T @ J)
                g = g + scale * (J.T @ r)
                loss = loss + scale * (r ** 2).sum()
        return loss, A, g

    def loss(self, p: torch.Tensor, y: torch.Tensor, alpha: torch.Tensor,
             sqrtw: torch.Tensor) -> torch.Tensor:
        """Total weighted loss of one member."""
        return sum(sqrtw[k] ** 2 * (v ** 2).mean() for k, v in enumerate(self.terms(p, y, alpha)))

    def term_losses(self, P: torch.Tensor, Y: torch.Tensor, alpha: torch.Tensor) -> list[torch.Tensor]:
        """Mean-square of every term for every member: list of (E,) tensors."""
        parts = vmap(self.terms)(P, Y, alpha)
        return [(v ** 2).mean(1) for v in parts]

    def raw_terms(self, P: torch.Tensor, Y: torch.Tensor, alpha: torch.Tensor) -> list[torch.Tensor]:
        """Residual vectors of every term for every member: list of (E, n_k)."""
        return list(vmap(self.terms)(P, Y, alpha))

    # ---------------------------------------------------------- arg packaging
    def pack(self, n_ens: int, Y=None, alpha=None, weights: Weights | list[Weights] | None = None
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-member arguments (Y, alpha, sqrt-weights) as tensors with leading axis E."""
        if Y is None:
            Y = torch.zeros(n_ens, 0, dtype=self.dtype, device=self.device)
        else:
            Y = self._t(Y)
            if Y.ndim == 1:
                Y = Y[None].expand(n_ens, -1).contiguous()
        if alpha is None:
            alpha = self.alpha_init
        alpha = self._t(np.broadcast_to(np.asarray(alpha, dtype=float), (n_ens,)).copy())
        if weights is None:
            weights = Weights()
        if isinstance(weights, Weights):
            weights = [weights] * n_ens
        W = self._t([[getattr(w, k) for k in self.term_names] for w in weights])
        return Y, alpha, torch.sqrt(W)

    # -------------------------------------------------------------- evaluation
    @torch.no_grad()
    def predict(self, P: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """u for every member at points Z -> (E, N)."""
        n_net = self.spec.n_params
        return vmap(lambda p: self.fields(p[:n_net], Z)[0])(P)

    @torch.no_grad()
    def field_error(self, P: torch.Tensor, alpha_true: float, n_points: int = 4096,
                    seed: int = 99) -> dict:
        """Relative L2 and max errors against the exact solution on Sobol test points."""
        d, T = self.dim, self.problem.T
        s = qmc.Sobol(d + 1, scramble=True, seed=seed).random(n_points)
        s[:, d] *= T
        exact = self._t(self.problem.mean(alpha_true, s[:, :d], s[:, d]))
        pred = self.predict(P, self._t(s))
        err = pred - exact[None]
        return {"l2_rel": (err.norm(dim=1) / exact.norm()).cpu().numpy(),
                "max_abs": err.abs().amax(dim=1).cpu().numpy()}
