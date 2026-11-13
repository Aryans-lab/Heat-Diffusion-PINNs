"""Checks of the closed-form theory against independent computations."""

import numpy as np
import pytest

from idealpinn.exact import (HeatBox, World, argmin_refined, beta_from_rho, beta_from_weights,
                             cramer_rao, delta_interval, objective_curves, rho_from_beta, run_study,
                             sensor_grid, simulate, weights_ratio_from_rho)
from idealpinn.exact.estimators import tree_level_field


@pytest.fixture(scope="module")
def setup():
    p = HeatBox(dim=1, n_modes=64)
    X, t = sensor_grid(1, 4, 10)
    return p, X, t


def test_mean_solves_heat_equation(setup):
    p, _, _ = setup
    a, h = 0.1, 1e-4
    x = np.array([[0.3], [0.62]])
    t = np.array([0.2, 0.7])
    u_t = (p.mean(a, x, t + h) - p.mean(a, x, t - h)) / (2 * h)
    u_xx = (p.mean(a, x + h, t) - 2 * p.mean(a, x, t) + p.mean(a, x - h, t)) / h ** 2
    assert np.allclose(u_t, a * u_xx, rtol=1e-4, atol=1e-6)
    assert np.allclose(p.mean(a, np.array([[0.0], [1.0]]), np.array([0.3, 0.3])), 0.0, atol=1e-12)
    assert np.allclose(p.mean(a, x, np.zeros(2)), p.initial_condition(x))


def test_mean_with_source_and_loss_solves_modified_equation():
    p = HeatBox(dim=1, n_modes=400)
    a, h, loss, src = 0.1, 1e-3, 0.3, 0.7
    x, t = np.array([[0.4]]), np.array([0.5])
    f = lambda xx, tt: p.mean(a, xx, tt, loss_rate=loss, source=src)
    u_t = (f(x, t + h) - f(x, t - h)) / (2 * h)
    u_xx = (f(x + h, t) - 2 * f(x, t) + f(x - h, t)) / h ** 2
    assert np.allclose(u_t, a * u_xx - loss * f(x, t) + src, rtol=2e-2)


def test_dmean_dalpha(setup):
    p, X, t = setup
    h = 1e-6
    fd = (p.mean(0.1 + h, X, t) - p.mean(0.1 - h, X, t)) / (2 * h)
    assert np.allclose(p.dmean_dalpha(0.1, X, t), fd, rtol=1e-6, atol=1e-9)


def test_kernel_is_symmetric_psd_and_diag_consistent(setup):
    p, X, t = setup
    C = p.kernel(0.1, X, t)
    assert np.allclose(C, C.T)
    assert np.linalg.eigvalsh(C).min() > -1e-10
    assert np.allclose(np.diag(C), p.kernel_diag(0.1, X, t), rtol=1e-10)


def test_kernel_matches_simulated_stochastic_heat_equation():
    """Covariance of exactly-simulated Ornstein-Uhlenbeck modes reproduces the kernel."""
    p = HeatBox(dim=1, n_modes=12)
    a = 0.1
    rng = np.random.default_rng(0)
    times = np.array([0.25, 0.6, 1.0])
    xs = np.array([[0.3], [0.7]])
    n = 200000
    rate = a * p.mu
    amp = np.zeros((n, len(rate)))
    samples = []
    prev = 0.0
    for tt in times:
        dt = tt - prev
        amp = amp * np.exp(-rate * dt) + rng.standard_normal(amp.shape) * np.sqrt(
            -np.expm1(-2 * rate * dt) / (2 * rate))
        samples.append(amp @ p.phi(xs).T)                       # (n, 2)
        prev = tt
    S = np.concatenate(samples, axis=1)                         # time-major ordering
    X = np.tile(xs, (len(times), 1))
    T = np.repeat(times, len(xs))
    C = p.kernel(a, X, T)
    emp = S.T @ S / n
    assert np.allclose(emp, C, atol=4 * np.sqrt(2.0 / n) * C.max())


def test_tree_level_equals_bruteforce_minimisation_over_the_field():
    """min_u [ |u(z_i)-y_i|^2/sigma^2 + beta int r^2 ]  computed by discretising u directly.

    The field is expanded in sine modes with time-gridded amplitudes and the
    quadratic problem is solved by dense least squares; the closed-form
    r^T (sigma^2 I + C/beta)^-1 r must agree up to the O(dt^2) quadrature error.
    """
    K, nt, T = 6, 240, 1.0
    p = HeatBox(dim=1, n_modes=K, ic_modes=K)
    X, t = sensor_grid(1, 3, 8, T)                             # times are multiples of T/8
    sigma, a0 = 0.02, 0.1
    y = simulate(p, X, t, sigma, a0, World("exact"), 1, np.random.default_rng(5))[0]
    pv = float(p.kernel_diag(a0, X, t).mean())
    dt = T / nt
    for rho, alpha in [(0.3, 0.1), (1.0, 0.085), (3.0, 0.12)]:
        beta = beta_from_rho(rho, sigma, pv)
        closed = objective_curves(p, X, t, y, sigma, [alpha], [beta]).tree[0, 0, 0]
        # unknowns: a_k(t_j), j = 1..nt (a_k(0) = c_k fixed) -> index k*nt + (j-1)
        lam = alpha * p.mu
        A = np.zeros((K * nt + len(t), K * nt))
        b = np.zeros(K * nt + len(t))
        w = np.sqrt(beta * dt)
        for k in range(K):
            for j in range(nt):                               # residual on [t_j, t_{j+1}]
                r = k * nt + j
                A[r, k * nt + j] = w * (1 / dt + lam[k] / 2)
                if j > 0:
                    A[r, k * nt + j - 1] = w * (-1 / dt + lam[k] / 2)
                else:
                    b[r] = -w * (-1 / dt + lam[k] / 2) * p.ic_coef[k]
        Phi = p.phi(X)
        for i in range(len(t)):
            j = int(round(t[i] / dt))
            for k in range(K):
                A[K * nt + i, k * nt + j - 1] = Phi[i, k] / sigma
            b[K * nt + i] = y[i] / sigma
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        brute = float(((A @ sol - b) ** 2).sum())
        assert brute == pytest.approx(closed, rel=1e-2), (rho, brute, closed)


def test_tree_level_field_interpolates_between_data_and_physics(setup):
    p, X, t = setup
    sigma = 0.02
    y = simulate(p, X, t, sigma, 0.1, World("exact"), 1, np.random.default_rng(2))[0]
    stiff = tree_level_field(p, X, t, y, sigma, 0.1, 1e12, X, t)
    loose = tree_level_field(p, X, t, y, sigma, 0.1, 1e-6, X, t)
    assert np.allclose(stiff, p.mean(0.1, X, t), atol=1e-6)         # physics wins
    assert np.allclose(loose, y, atol=1e-3)                          # data are interpolated


def test_conversions_roundtrip():
    assert rho_from_beta(beta_from_rho(0.7, 0.02, 0.8), 0.02, 0.8) == pytest.approx(0.7)
    ratio = weights_ratio_from_rho(0.5, 100, 1.0, 0.8)
    beta = beta_from_weights(1.0, ratio, 100, 0.02, 1.0)
    assert rho_from_beta(beta, 0.02, 0.8) == pytest.approx(0.5)


def test_argmin_and_interval_on_a_parabola():
    grid = np.linspace(-3, 3, 61)
    J = ((grid[:, None] - np.array([0.123, -1.01])) / np.array([0.5, 0.8])) ** 2
    est, fmin, edge = argmin_refined(grid, J)
    lo, hi, open_ = delta_interval(grid, J)
    assert np.allclose(est, [0.123, -1.01], atol=1e-10) and not edge.any() and not open_.any()
    assert np.allclose(fmin, 0.0, atol=1e-10)
    assert np.allclose(hi - lo, [1.0, 1.6], atol=2e-2)


def test_limits_and_calibration_in_the_exact_world(setup):
    p, X, t = setup
    res = run_study(p, X, t, 0.02, 0.1, World("exact"), [0.01, 1.0], n_experiments=500, seed=1)
    e = res["estimators"]
    # hard-constraint limit: tiny rho reproduces the MLE, which is efficient and calibrated
    assert e["tree@0.01"]["bias"] == pytest.approx(e["mle"]["bias"], abs=0.02)
    assert e["mle"]["sd"] == pytest.approx(1.0, abs=0.12)
    assert abs(e["mle"]["bias"]) < 0.15
    assert 0.6 < e["mle"]["coverage"] < 0.76
    # loose physics: the ideal PINN is biased towards small diffusivity, one-loop the other way
    assert e["tree@1"]["bias"] < -0.4 and e["oneloop@1"]["bias"] > 0.2
    assert e["tree@1"]["sd"] > 1.3
    # learning the physics precision recovers (almost) the MLE
    assert e["oneloop_learned"]["rmse"] < 1.25 * e["mle"]["rmse"]


def test_one_loop_is_efficient_when_model_error_is_real(setup):
    p, X, t = setup
    res = run_study(p, X, t, 0.02, 0.1, World("stochastic", rho_true=1.0), [1.0],
                    n_experiments=500, seed=2)
    e, m = res["estimators"], res["meta"]
    crb_known = cramer_rao(p, X, t, 0.02, 0.1, beta=m["betas"][0]) / m["crb"]
    assert e["oneloop@1"]["sd"] == pytest.approx(crb_known, rel=0.12)
    assert abs(e["oneloop@1"]["bias"]) < 0.5
    assert 0.6 < e["oneloop@1"]["coverage"] < 0.76
    assert e["mle"]["coverage"] < 0.35                               # overconfident
    assert e["tree@1"]["bias"] < e["oneloop@1"]["bias"] - 0.4       # biased even with the right rho
