import tensorflow as tf
import numpy as np

def generate_1d_data(n_points=60000, L=1.0, T=1.0):
    """
    Generate training data for 1D heat equation (domain: -L to +L)
    Returns: 
        dict: {import numpy as np
import math

x = float(input("Enter x: "))
acc_needed = float(input("Enter accuracy: "))
true_value = np.sin(x)
approx_value = 0
i = 0

accuracy = abs((true_value - approx_value) / true_value)

while accuracy > acc_needed:  # keep looping until error is small enough
    term = ((-1)**i) * (x**(2*i+1)) / math.factorial(2*i+1)
    approx_value += term
    i += 1
    accuracy = abs((true_value - approx_value) / true_value)

print("To achieve this accuracy, it needs to be computed for", i, "terms.")

            'collocation': interior points [x, t],
            'boundary': boundary points [x, t],
            'initial': initial condition points [x, t=0]
        }
    """
    # Each set gets 1/3 of total points
    n_each = n_points // 3

    # Interior collocation points
    x_coll = tf.random.uniform((n_each, 1), -L, L)
    t_coll = tf.random.uniform((n_each, 1), 0, T)
    coll_points = tf.concat([x_coll, t_coll], axis=1)

    # Boundary points (x = -L and x = +L)
    n_bc = n_each
    x_left = -L * tf.ones((n_bc // 2, 1), dtype=tf.float32)   # x = -L
    x_right = L * tf.ones((n_bc // 2, 1), dtype=tf.float32)   # x = +L
    t_bc = tf.random.uniform((n_bc, 1), 0, T)                 # Random times
    bc_points = tf.concat([tf.concat([x_left, x_right], axis=0), t_bc], axis=1)
    bc_points = tf.random.shuffle(bc_points)

    # Initial condition points (t = 0)
    x_ic = tf.random.uniform((n_each, 1), -L, L)
    ic_points = tf.concat([x_ic, tf.zeros((n_each, 1))], axis=1)

    return {
        'collocation': coll_points,
        'boundary': bc_points,
        'initial': ic_points
    }

def gaussian_initial_condition(x, center=0.0, amplitude=1.0, width=0.1):
    """Initial temperature distribution (centered at 0 for symmetric domain)"""
    return amplitude * tf.exp(-(x - center)**2 / (2 * width**2))
