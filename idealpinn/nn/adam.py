"""Batched Adam training with the standard first-order PINN toolbox.

Implements, for a whole ensemble at once:

* warm-up + exponential learning-rate decay,
* periodic re-sampling of collocation points,
* fixed loss weights or gradient-norm balancing (Wang, Teng & Perdikaris 2021),
* temporal-causality weighting of the PDE residual (Wang, Sankaran & Perdikaris 2022).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import torch

from .heat import HeatPINN


@dataclass
class AdamConfig:
    steps: int = 20000
    lr: float = 1e-3
    warmup: int = 500
    decay_rate: float = 0.9
    decay_steps: int = 2000
    resample_every: int = 0          # 0 keeps the collocation set fixed
    weighting: str = "fixed"         # "fixed" | "gradnorm"
    gradnorm_every: int = 100
    gradnorm_momentum: float = 0.9
    causal_eps: float = 0.0          # 0 disables causal weighting
    causal_bins: int = 16
    log_every: int = 100


def _lr(cfg: AdamConfig, step: int) -> float:
    warm = min(1.0, (step + 1) / max(cfg.warmup, 1))
    return cfg.lr * warm * cfg.decay_rate ** (step / cfg.decay_steps)


def adam_fit(pinn: HeatPINN, P0: torch.Tensor, args: tuple, cfg: AdamConfig,
             callback: Callable | None = None, verbose: bool = False) -> tuple[torch.Tensor, dict]:
    """Train all members with Adam.  ``args`` comes from :meth:`HeatPINN.pack`.

    Returns the parameters and a history dict with per-member term losses, the
    effective weights (relevant for gradient-norm balancing) and wall time.
    """
    Y, alpha, sqrtw = args
    W = (sqrtw ** 2).clone()                                  # (E, n_terms)
    P = P0.clone().requires_grad_(True)
    opt = torch.optim.Adam([P], lr=cfg.lr)
    names = pinn.term_names
    i_res = names.index("res")
    hist = {"step": [], "terms": [], "weights": [], "time": []}
    t0 = time.time()

    def losses():
        parts = pinn.raw_terms(P, Y, alpha)
        out = []
        for k, v in enumerate(parts):
            if k == i_res and cfg.causal_eps > 0:
                tt = pinn.Zc[:, pinn.dim] / pinn.problem.T
                bins = torch.clamp((tt * cfg.causal_bins).long(), max=cfg.causal_bins - 1)
                onehot = torch.nn.functional.one_hot(bins, cfg.causal_bins).to(v.dtype)
                Lb = (v ** 2) @ onehot / onehot.sum(0).clamp_min(1.0)        # (E, bins)
                cum = torch.cumsum(Lb, dim=1) - Lb
                wc = torch.exp(-cfg.causal_eps * cum).detach()
                out.append((wc * Lb).mean(1))
            else:
                out.append((v ** 2).mean(1))
        return out

    for step in range(cfg.steps):
        if cfg.resample_every and step > 0 and step % cfg.resample_every == 0:
            pinn.resample(1000 + step)
        for g in opt.param_groups:
            g["lr"] = _lr(cfg, step)
        L = losses()
        if cfg.weighting == "gradnorm" and step % cfg.gradnorm_every == 0:
            norms = []
            for Lk in L:
                gk, = torch.autograd.grad(Lk.sum(), P, retain_graph=True)
                norms.append(gk.norm(dim=1))
            norms = torch.stack(norms, dim=1).clamp_min(1e-30)               # (E, n_terms)
            new = norms.sum(1, keepdim=True) / norms
            W = cfg.gradnorm_momentum * W + (1 - cfg.gradnorm_momentum) * new if step > 0 else new
        total = sum((W[:, k] * Lk).sum() for k, Lk in enumerate(L))
        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()
        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            hist["step"].append(step)
            hist["terms"].append(torch.stack([l.detach() for l in L], dim=1).cpu())
            hist["weights"].append(W.detach().cpu().clone())
            hist["time"].append(time.time() - t0)
            if callback is not None:
                callback(step, P.detach(), L)
            if verbose and step % (cfg.log_every * 10) == 0:
                msg = " ".join(f"{n}={l.median().item():.2e}" for n, l in zip(names, L))
                print(f"  Adam {step:6d}  {msg}  [{time.time() - t0:.0f}s]", flush=True)
    hist["terms"] = torch.stack(hist["terms"])
    hist["weights"] = torch.stack(hist["weights"])
    hist["names"] = names
    return P.detach(), hist
