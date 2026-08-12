"""Finite-difference checks for the automatic-differentiation losses.

These tests verify that the derivatives used by the physics losses are
correct. A wrong second derivative silently corrupts every training run,
so this is the single most valuable guard in the project.
"""

import numpy as np
import pytest
import tensorflow as tf

from models.pinn import build_pinn
from training.losses import (
    heat_1d_loss,
    heat_2d_loss,
    heat_3d_loss,
    compute_second_order_derivatives,
)


def _build_fp64_model(dim):
    """Small double-precision network so finite differences are well-conditioned."""
    tf.keras.utils.set_random_seed(11)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(dim,), dtype=tf.float64))
    for _ in range(2):
        model.add(tf.keras.layers.Dense(12, activation="tanh", dtype=tf.float64))
    model.add(tf.keras.layers.Dense(1, dtype=tf.float64))
    return model


def _finite_difference_derivatives(model, inputs, eps=1e-5):
    """Central finite differences of the model output at the given inputs."""
    x = np.asarray(inputs, dtype=np.float64)
    n, d = x.shape

    def f(xx):
        return model(tf.constant(xx, dtype=tf.float64)).numpy()[:, 0]

    first = np.empty((n, d))
    second = np.empty((n, d))
    for i in range(d):
        xp = x.copy()
        xm = x.copy()
        xp[:, i] += eps
        xm[:, i] -= eps
        first[:, i] = (f(xp) - f(xm)) / (2 * eps)

        xpp = x.copy()
        xmm = x.copy()
        xpp[:, i] += eps
        xmm[:, i] -= eps
        second[:, i] = (f(xpp) - 2 * f(x) + f(xmm)) / eps ** 2
    return first, second


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_second_order_derivatives_match_finite_difference(dim):
    model = _build_fp64_model(dim + 1)
    inputs = np.random.default_rng(0).uniform(-0.8, 0.8, size=(16, dim + 1))

    _, first, second = compute_second_order_derivatives(model, tf.constant(inputs))
    fd_first, fd_second = _finite_difference_derivatives(model, inputs)

    np.testing.assert_allclose(first.numpy(), fd_first, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(second.numpy(), fd_second, rtol=1e-4, atol=1e-5)


def _residual_from_helper(model, inputs, alpha):
    _, first, second = compute_second_order_derivatives(model, inputs)
    laplacian = tf.reduce_sum(second[:, :-1], axis=1)
    residual = first[:, -1] - alpha * laplacian
    return tf.reduce_mean(tf.square(residual))


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_helper_residual_matches_physics_loss(dim):
    """The generic derivative helper must reproduce each physics loss exactly."""
    tf.keras.utils.set_random_seed(3)
    model = build_pinn(input_dim=dim + 1, hidden_units=12, num_layers=2)
    inputs = tf.random.uniform((32, dim + 1), minval=-0.8, maxval=0.8)

    loss_fn = {1: heat_1d_loss, 2: heat_2d_loss, 3: heat_3d_loss}[dim]
    expected = loss_fn(model, inputs, alpha=0.1)
    actual = _residual_from_helper(model, inputs, alpha=0.1)

    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-4, atol=1e-6)


def test_heat_losses_return_finite_nonnegative_values():
    for dim, loss_fn in [(1, heat_1d_loss), (2, heat_2d_loss), (3, heat_3d_loss)]:
        tf.keras.utils.set_random_seed(5)
        model = build_pinn(input_dim=dim + 1, hidden_units=8, num_layers=1)
        inputs = tf.random.uniform((64, dim + 1), minval=-1.0, maxval=1.0)
        loss = loss_fn(model, inputs, alpha=0.1)
        assert float(loss) >= 0.0
        assert np.isfinite(float(loss))
