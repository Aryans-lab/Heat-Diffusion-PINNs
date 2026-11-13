"""Checks of the neural half: derivatives, Gauss-Newton system, training."""

import numpy as np
import pytest
import torch

from idealpinn.exact import HeatBox, World, sensor_grid, simulate
from idealpinn.nn import AdamConfig, HeatPINN, NetSpec, Weights, adam_fit, forward_jet, init_params, lm_fit

torch.set_num_threads(2)


@pytest.mark.parametrize("fourier,rwf", [(0, False), (8, True)])
def test_forward_jet_matches_autograd(fourier, rwf):
    spec = NetSpec(in_dim=3, width=10, depth=2, fourier_features=fourier, rwf=rwf)
    p = init_params(spec, 1, 0, torch.float64, "cpu")[0]
    z = torch.rand(5, 3, dtype=torch.float64)
    u, du, d2u = forward_jet(spec, p, z)
    f = lambda zz: forward_jet(spec, p, zz[None])[0][0]
    for i in range(5):
        g = torch.autograd.functional.jacobian(f, z[i])
        H = torch.autograd.functional.hessian(f, z[i])
        assert torch.allclose(du[:, i], g, atol=1e-10)
        assert torch.allclose(d2u[:, i], torch.diagonal(H)[:d2u.shape[0]], atol=1e-9)   # spatial second derivatives


def test_hard_constraints_hold_exactly():
    p = HeatBox(dim=2, n_modes=16)
    pinn = HeatPINN(p, NetSpec(in_dim=3, width=8, depth=2), constraint="ic_bc", n_colloc=64, device="cpu")
    P = pinn.init(2, 0)
    x = np.random.default_rng(0).random((20, 2))
    Z0 = torch.tensor(np.c_[x, np.zeros(20)])
    assert torch.allclose(pinn.predict(P, Z0)[0], torch.tensor(p.initial_condition(x)), atol=1e-12)
    xb = x.copy()
    xb[:10, 0] = 0.0
    xb[10:, 1] = 1.0
    Zb = torch.tensor(np.c_[xb, np.full(20, 0.4)])
    assert pinn.predict(P, Zb).abs().max() < 1e-12


def test_gauss_newton_system_is_consistent_with_the_loss():
    p = HeatBox(dim=1, n_modes=32)
    X, t = sensor_grid(1, 3, 4)
    pinn = HeatPINN(p, NetSpec(in_dim=2, width=6, depth=2), n_colloc=64, sensors=(X, t),
                    learn_alpha=True, device="cpu")
    y = simulate(p, X, t, 0.02, 0.1, World("exact"), 1, np.random.default_rng(0))
    Y, alpha, sqrtw = pinn.pack(1, Y=y, weights=Weights(data=1.0, res=0.3))
    P = pinn.init(1, 0)[0].clone().requires_grad_(True)
    loss, A, g = pinn.system(P, Y[0], alpha[0], sqrtw[0])
    L = pinn.loss(P, Y[0], alpha[0], sqrtw[0])
    grad, = torch.autograd.grad(L, P)
    assert torch.allclose(loss, L, rtol=1e-12)
    assert torch.allclose(2 * g, grad, rtol=1e-8, atol=1e-12)
    assert torch.allclose(A, A.T) and torch.linalg.eigvalsh(A).min() > -1e-10


@pytest.mark.slow
def test_lm_solves_the_forward_problem():
    p = HeatBox(dim=1, n_modes=32)
    pinn = HeatPINN(p, NetSpec(in_dim=2, width=16, depth=2), n_colloc=512, device="cpu")
    args = pinn.pack(1, alpha=0.1)
    P, info = lm_fit(pinn.system, pinn.loss, pinn.init(1, 0), args, n_iter=120)
    assert info["loss"][-1].max() < 1e-3 * info["loss"][0].min()
    assert pinn.field_error(P, 0.1)["l2_rel"].max() < 2e-2


@pytest.mark.slow
def test_adam_reduces_the_loss():
    p = HeatBox(dim=1, n_modes=32)
    pinn = HeatPINN(p, NetSpec(in_dim=2, width=16, depth=2), n_colloc=256, device="cpu")
    args = pinn.pack(2, alpha=0.1)
    P, hist = adam_fit(pinn, pinn.init(2, 0), args, AdamConfig(steps=500, lr=5e-3, warmup=50, log_every=50))
    assert (hist["terms"][-1] < 0.5 * hist["terms"][0]).all()
