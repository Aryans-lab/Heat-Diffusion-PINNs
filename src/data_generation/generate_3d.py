import tensorflow as tf
import numpy as np

def generate_3d_cube(n_points=90000, L=1.0, T=1.0):
    """Generate 3D training data for cubic domain [-L, L]^3."""
    n_each = n_points // 3  # divide equally: collocation, boundary, initial

    # Collocation points: inside volume for PDE loss
    x_coll = tf.random.uniform((n_each, 1), -L, L)
    y_coll = tf.random.uniform((n_each, 1), -L, L)
    z_coll = tf.random.uniform((n_each, 1), -L, L)
    t_coll = tf.random.uniform((n_each, 1), 0.0, T)

    # Boundary points: all 6 faces of cube
    n_face = n_each // 6
    t_bc = tf.random.uniform((n_face, 1), 0.0, T)

    boundary = tf.concat([
        tf.concat([tf.fill((n_face, 1), -L), tf.random.uniform((n_face, 1), -L, L),
                   tf.random.uniform((n_face, 1), -L, L), t_bc], axis=1),  # x = -L
        tf.concat([tf.fill((n_face, 1),  L), tf.random.uniform((n_face, 1), -L, L),
                   tf.random.uniform((n_face, 1), -L, L), t_bc], axis=1),  # x = +L
        tf.concat([tf.random.uniform((n_face, 1), -L, L), tf.fill((n_face, 1), -L),
                   tf.random.uniform((n_face, 1), -L, L), t_bc], axis=1),  # y = -L
        tf.concat([tf.random.uniform((n_face, 1), -L, L), tf.fill((n_face, 1),  L),
                   tf.random.uniform((n_face, 1), -L, L), t_bc], axis=1),  # y = +L
        tf.concat([tf.random.uniform((n_face, 1), -L, L), tf.random.uniform((n_face, 1), -L, L),
                   tf.fill((n_face, 1), -L), t_bc], axis=1),  # z = -L
        tf.concat([tf.random.uniform((n_face, 1), -L, L), tf.random.uniform((n_face, 1), -L, L),
                   tf.fill((n_face, 1),  L), t_bc], axis=1),  # z = +L
    ], axis=0)

    # Initial condition points (t = 0)
    x_ic = tf.random.uniform((n_each, 1), -L, L)
    y_ic = tf.random.uniform((n_each, 1), -L, L)
    z_ic = tf.random.uniform((n_each, 1), -L, L)
    t_ic = tf.zeros((n_each, 1))

    return {
        'collocation': tf.concat([x_coll, y_coll, z_coll, t_coll], axis=1),
        'boundary': boundary,
        'initial': tf.concat([x_ic, y_ic, z_ic, t_ic], axis=1),
    }

def gaussian_initial_condition_3d(x, y, z, center=(0.0, 0.0, 0.0), amplitude=1.5, width=0.25):
    """Initial Gaussian temperature distribution in 3D."""
    return amplitude * tf.exp(-(
        (x - center[0])**2 +
        (y - center[1])**2 +
        (z - center[2])**2
    ) / (2 * width**2))
