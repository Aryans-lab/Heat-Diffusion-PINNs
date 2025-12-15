import tensorflow as tf
import numpy as np

def generate_2d_square(n_points=60000, L=1.0, T=1.0):
    """Generate 2D training data for square domain [-L, L]."""
    n_each = n_points // 3

    # Collocation points
    x_coll = tf.random.uniform((n_each, 1), -L, L)
    y_coll = tf.random.uniform((n_each, 1), -L, L)
    t_coll = tf.random.uniform((n_each, 1), 0.0, T)

    # Boundary points (4 edges: x = ±L or y = ±L)
    n_edge = n_each // 4
    t_bc = tf.random.uniform((n_edge, 1), 0.0, T)
    boundary = tf.concat([
        tf.concat([tf.fill((n_edge, 1), -L), tf.random.uniform((n_edge, 1), -L, L), t_bc], axis=1),  # Left
        tf.concat([tf.fill((n_edge, 1),  L), tf.random.uniform((n_edge, 1), -L, L), t_bc], axis=1),  # Right
        tf.concat([tf.random.uniform((n_edge, 1), -L, L), tf.fill((n_edge, 1), -L), t_bc], axis=1),  # Bottom
        tf.concat([tf.random.uniform((n_edge, 1), -L, L), tf.fill((n_edge, 1),  L), t_bc], axis=1),  # Top
    ], axis=0)

    # Initial condition points (t = 0)
    x_ic = tf.random.uniform((n_each, 1), -L, L)
    y_ic = tf.random.uniform((n_each, 1), -L, L)
    t_ic = tf.zeros((n_each, 1))

    return {
        'collocation': tf.concat([x_coll, y_coll, t_coll], axis=1),
        'boundary': boundary,
        'initial': tf.concat([x_ic, y_ic, t_ic], axis=1),
    }

def gaussian_initial_condition_2d(x, y, center=(0.0, 0.0), amplitude=1.5, width=0.25):
    """Initial Gaussian temperature distribution centered at (0,0)."""
    return amplitude * tf.exp(-((x-center[0])**2 + (y-center[1])**2) / (2 * width**2))
