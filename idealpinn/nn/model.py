"""Functional MLP with exact derivative ("jet") propagation.

Design choices
--------------
* **Functional**: the network is a pure function ``forward_jet(spec, p, Z)`` of
  a flat parameter vector ``p``.  This makes it trivial to train a whole
  *ensemble* of independent networks in parallel with ``torch.func.vmap`` and
  to build exact Jacobians for Gauss-Newton / Levenberg-Marquardt.
* **Jets instead of nested autograd**: u, du/dz_j and d2u/dz_j^2 are propagated
  forward through the layers analytically (a hand-written second-order
  Taylor mode).  This is exact, several times cheaper than nested reverse
  mode, and leaves a single differentiation w.r.t. the parameters.
* **Fourier features** (Tancik et al. 2020; Wang et al. 2021) and **random
  weight factorisation** (Wang et al. 2022) are optional.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import math
import torch


@dataclass
class NetSpec:
    in_dim: int                      # number of coordinates (space dims + time)
    width: int = 32
    depth: int = 3                   # number of hidden layers
    fourier_features: int = 0        # number of random frequencies (0 disables)
    fourier_scale: float = 2.0       # std of the random frequencies
    include_raw: bool = True         # concatenate raw coordinates to the features
    rwf: bool = False                # random weight factorisation
    rwf_mean: float = 0.5
    rwf_std: float = 0.1
    second_dims: tuple[int, ...] | None = None   # coordinates needing d2/dz^2 (default: all but last)
    feature_seed: int = 1234
    B: torch.Tensor | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.second_dims is None:
            self.second_dims = tuple(range(self.in_dim - 1))
        if self.fourier_features > 0 and self.B is None:
            g = torch.Generator().manual_seed(self.feature_seed)
            self.B = self.fourier_scale * torch.randn(self.fourier_features, self.in_dim,
                                                      generator=g, dtype=torch.float64)

    # ------------------------------------------------------------ bookkeeping
    @property
    def feature_dim(self) -> int:
        if self.fourier_features == 0:
            return self.in_dim
        return 2 * self.fourier_features + (self.in_dim if self.include_raw else 0)

    @property
    def layer_sizes(self) -> list[tuple[int, int]]:
        sizes = [self.feature_dim] + [self.width] * self.depth + [1]
        return [(sizes[i + 1], sizes[i]) for i in range(len(sizes) - 1)]

    @property
    def n_params(self) -> int:
        n = 0
        for out, inp in self.layer_sizes:
            n += out * inp + out + (out if self.rwf else 0)
        return n

    def unflatten(self, p: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Split a flat vector into [(W, b), ...] (W already includes the RWF scale)."""
        layers = []
        i = 0
        for out, inp in self.layer_sizes:
            W = p[i:i + out * inp].reshape(out, inp)
            i += out * inp
            if self.rwf:
                s = p[i:i + out]
                i += out
                W = torch.exp(s)[:, None] * W
            b = p[i:i + out]
            i += out
            layers.append((W, b))
        return layers


def init_params(spec: NetSpec, n_ens: int, seed: int = 0, dtype=torch.float64,
                device: str | torch.device = "cpu") -> torch.Tensor:
    """Glorot-normal initialisation of ``n_ens`` independent networks -> (n_ens, P)."""
    g = torch.Generator().manual_seed(seed)
    chunks = []
    for out, inp in spec.layer_sizes:
        std = math.sqrt(2.0 / (inp + out))
        W = std * torch.randn(n_ens, out, inp, generator=g, dtype=torch.float64)
        if spec.rwf:
            s = spec.rwf_mean + spec.rwf_std * torch.randn(n_ens, out, generator=g, dtype=torch.float64)
            W = W / torch.exp(s)[:, :, None]
            chunks += [W.reshape(n_ens, -1), s]
        else:
            chunks.append(W.reshape(n_ens, -1))
        chunks.append(torch.zeros(n_ens, out, dtype=torch.float64))
    return torch.cat(chunks, dim=1).to(dtype=dtype, device=device)


def forward_jet(spec: NetSpec, p: torch.Tensor, Z: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate one network and its input derivatives.

    Parameters
    ----------
    p : (P,) flat parameters of a single network.
    Z : (N, D) input coordinates.

    Returns
    -------
    u   : (N,)     network output
    du  : (D, N)   first derivatives w.r.t. every coordinate
    d2u : (S, N)   second derivatives w.r.t. the coordinates in ``spec.second_dims``
    """
    N, D = Z.shape
    sd = list(spec.second_dims)
    eye = torch.eye(D, dtype=Z.dtype, device=Z.device)
    h = Z
    dh = eye[:, None, :].expand(D, N, D)                     # (D, N, D)
    d2h = torch.zeros(len(sd), N, D, dtype=Z.dtype, device=Z.device)

    if spec.fourier_features > 0:
        B = spec.B.to(dtype=Z.dtype, device=Z.device)        # (F, D)
        proj = Z @ B.T                                       # (N, F)
        s, c = torch.sin(proj), torch.cos(proj)
        Bj = B.T[:, None, :]                                 # (D, 1, F)
        dfeat = torch.cat([c[None] * Bj, -s[None] * Bj], dim=-1)             # (D, N, 2F)
        Bs = Bj[sd] ** 2
        d2feat = torch.cat([-s[None] * Bs, -c[None] * Bs], dim=-1)           # (S, N, 2F)
        feat = torch.cat([s, c], dim=-1)
        if spec.include_raw:
            h = torch.cat([h, feat], dim=-1)
            dh = torch.cat([dh, dfeat], dim=-1)
            d2h = torch.cat([d2h, d2feat], dim=-1)
        else:
            h, dh, d2h = feat, dfeat, d2feat

    layers = spec.unflatten(p)
    for W, b in layers[:-1]:
        z = h @ W.T + b
        dz = dh @ W.T
        d2z = d2h @ W.T
        a = torch.tanh(z)
        g1 = 1.0 - a * a
        h = a
        dh = g1 * dz
        d2h = g1 * d2z - 2.0 * a * g1 * dz[sd] ** 2
    W, b = layers[-1]
    u = (h @ W.T + b)[:, 0]
    du = (dh @ W.T)[..., 0]
    d2u = (d2h @ W.T)[..., 0]
    return u, du, d2u
