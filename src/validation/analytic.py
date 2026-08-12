"""Analytic solutions of the heat equation with zero Dirichlet boundary
conditions on the hypercube [-L, L]^d.

For a separable Gaussian initial condition

    u(x_1, ..., x_d, 0) = A * exp(-(x_1^2 + ... + x_d^2) / (2 sigma^2))
                        = prod_i g(x_i),

separation of variables gives the exact solution

    u(x_1, ..., x_d, t) = prod_i s(x_i, t),

where, with k_n = n*pi / (2L),

    s(x, t) = sum_{n>=1} B_n * exp(-alpha k_n^2 t) * sin(k_n (x + L))

and B_n are the sine-series coefficients of g on [-L, L],

    B_n = (1/L) * int_{-L}^{L} g(x) * sin(k_n (x + L)) dx.
"""

import numpy as np


def _sine_coeffs(g, L, n_terms, n_grid=5001):
    """Sine-series coefficients B_n of a function g on [-L, L]."""
    xs = np.linspace(-L, L, n_grid)
    k_n = np.pi * np.arange(1, n_terms + 1) / (2 * L)
    integrand = g(xs)[:, None] * np.sin(k_n[None, :] * (xs[:, None] + L))
    return np.trapezoid(integrand, xs, axis=0) / L


def gaussian_factor(amplitude, width, dim):
    """Return g(x) so that prod_i g(x_i) equals a d-dimensional Gaussian.

    The per-axis amplitude A^(1/d) makes the product reproduce the total
    amplitude exactly.
    """
    axis_amplitude = amplitude ** (1.0 / dim)
    return lambda x: axis_amplitude * np.exp(-(x ** 2) / (2 * width ** 2))


def _factor_1d(g, L, alpha, xs, t, n_terms=200):
    """Evaluate the 1D factor s(x, t) of the separable solution."""
    coeffs = _sine_coeffs(g, L, n_terms)
    k_n = np.pi * np.arange(1, n_terms + 1) / (2 * L)
    basis = np.sin(k_n[:, None] * (xs[None, :] + L))
    return (coeffs * np.exp(-alpha * k_n ** 2 * t)) @ basis


def ic_field(dim, g, xs):
    """Evaluate the separable initial condition on a uniform grid of xs."""
    s = g(xs)
    if dim == 1:
        return s
    if dim == 2:
        return s[:, None] * s[None, :]
    return s[:, None, None] * s[None, :, None] * s[None, None, :]


def exact_solution(dim, g, L, alpha, t, xs, n_terms=200):
    """Exact heat-equation solution at time t on a uniform grid of xs.

    Args:
        dim: space dimension (1, 2 or 3).
        g: 1D factor of the separable initial condition (see gaussian_factor).
        L: half-domain length; zero Dirichlet BCs on [-L, L]^d.
        alpha: thermal diffusivity.
        t: time at which to evaluate.
        xs: 1D grid of coordinates in [-L, L].

    Returns:
        Array of shape (n,) for dim 1, (n, n) for dim 2, (n, n, n) for dim 3.
    """
    s = _factor_1d(g, L, alpha, xs, t, n_terms)
    if dim == 1:
        return s
    if dim == 2:
        return s[:, None] * s[None, :]
    return s[:, None, None] * s[None, :, None] * s[None, None, :]
