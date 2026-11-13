"""Batched Levenberg-Marquardt for PINN least-squares problems.

A PINN loss is a sum of squares, loss(p) = |r(p)|^2, so for the small networks
that suffice for low-dimensional PDEs the Gauss-Newton matrix J^T J can be
formed exactly.  Levenberg-Marquardt then converges orders of magnitude
further than first-order optimisers (it is the dense, exact-Jacobian limit of
the natural-gradient / Gauss-Newton family used by recent high-accuracy PINN
work).  All ensemble members are advanced simultaneously; each one carries its
own damping parameter.
"""

from __future__ import annotations

import time
from typing import Callable

import torch
from torch.func import vmap


def lm_fit(system_fn: Callable, loss_fn: Callable, p0: torch.Tensor, args: tuple = (),
           n_iter: int = 300, mu0: float = 1e-2, mu_min: float = 1e-14, mu_max: float = 1e10,
           tol: float = 1e-12, patience: int = 8, callback: Callable | None = None,
           verbose: bool = False, frozen: torch.Tensor | None = None,
           freeze_iters: int = 0) -> tuple[torch.Tensor, dict]:
    """Minimise a sum of squares independently for every ensemble member.

    Parameters
    ----------
    system_fn : single-member function ``(p, *args) -> (loss, A, g)`` with
        loss = |r|^2, A = J^T J and g = J^T r  (J = dr/dp).
    loss_fn : single-member function ``(p, *args) -> loss``.
        Both are vmapped over the leading ensemble axis of ``p0`` and ``args``.
    p0 : (E, P) initial parameters.
    callback : optional ``callback(iteration, p, loss)`` called after every iteration.
    frozen, freeze_iters : boolean mask (P,) of parameters held fixed during the
        first ``freeze_iters`` iterations (used to warm up the field before the
        PDE parameter is released).

    Returns
    -------
    p : (E, P) optimised parameters
    info : dict with ``loss`` history (iters+1, E), wall ``time`` and iteration count.
    """
    p = p0.clone()
    E = p.shape[0]
    system_b = vmap(system_fn)
    loss_b = vmap(loss_fn)

    def system(q, it_next=1):
        loss, A, g = system_b(q, *args)
        if frozen is not None and it_next <= freeze_iters:
            keep = (~frozen).to(A.dtype)
            A = A * keep[None, :, None] * keep[None, None, :] + torch.diag_embed((1 - keep)[None].expand(E, -1))
            g = g * keep[None]
        return loss, A, g

    t0 = time.time()
    loss, A, g = system(p)
    mu = torch.full((E,), mu0, dtype=p.dtype, device=p.device)
    hist = [loss.detach().cpu()]
    stall = torch.zeros(E, dtype=torch.long, device=p.device)
    it = 0
    for it in range(1, n_iter + 1):
        D = torch.diagonal(A, dim1=1, dim2=2)
        D = torch.maximum(D, 1e-8 * D.amax(dim=1, keepdim=True)).clamp_min(1e-300)
        L, info = torch.linalg.cholesky_ex(A + torch.diag_embed(mu[:, None] * D))
        bad = info != 0                                   # not positive definite at this damping
        if bad.any():
            L = torch.where(bad[:, None, None], torch.diag_embed(torch.ones_like(D)), L)
        delta = -torch.cholesky_solve(g[..., None], L)[..., 0]
        p_new = p + delta
        loss_new = loss_b(p_new, *args)
        pred = -(2.0 * (delta * g).sum(1) + (delta[:, None, :] @ A @ delta[:, :, None])[:, 0, 0])
        gain = (loss - loss_new) / pred.clamp_min(1e-300)
        accept = torch.isfinite(loss_new) & (loss_new < loss) & ~bad
        rel = (loss - loss_new) / loss.clamp_min(1e-300)
        small = accept & (rel < tol)
        stall = torch.where(small, stall + 1, torch.where(accept, torch.zeros_like(stall), stall))
        mu = torch.where(accept & (gain > 0.75), mu / 3.0, mu)
        mu = torch.where(accept & (gain < 0.25), mu * 2.0, mu)
        mu = torch.where(~accept, mu * 4.0, mu)
        done = (stall >= patience) | (mu >= mu_max)
        if frozen is not None and it < freeze_iters:
            done = torch.zeros_like(done)
        mu = mu.clamp(mu_min, mu_max)
        release = frozen is not None and it == freeze_iters
        if accept.any() or release:
            p = torch.where(accept[:, None], p_new, p)
            loss, A, g = system(p, it + 1)
        if release:                                       # new problem: restart damping and stalls
            mu = torch.full_like(mu, mu0)
            stall = torch.zeros_like(stall)
            done = torch.zeros_like(done)
        hist.append(loss.detach().cpu())
        if callback is not None:
            callback(it, p, loss)
        if verbose and (it % 10 == 0 or it == 1):
            print(f"  LM {it:4d}  loss[min/med/max]= {loss.min():.3e} {loss.median():.3e} "
                  f"{loss.max():.3e}  mu_med={mu.median():.1e}  [{time.time() - t0:.0f}s]", flush=True)
        if bool(done.all()):
            break
    return p, {"loss": torch.stack(hist), "time": time.time() - t0, "iters": it}
