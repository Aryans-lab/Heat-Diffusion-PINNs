import tensorflow as tf

@tf.function
def compute_derivatives(model, inputs):
    # inputs: [x, t]
    inputs = tf.convert_to_tensor(inputs)
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(inputs)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(inputs)
            u = model(inputs)  # u(x, t)
        u_x = tape1.gradient(u, inputs)[:, 0:1]  # ∂u/∂x
        u_t = tape1.gradient(u, inputs)[:, 1:2]  # ∂u/∂t
    u_xx = tape2.gradient(u_x, inputs)[:, 0:1]  # ∂²u/∂x²
    return u, u_t, u_xx

def heat_1d_loss(model, inputs, alpha=0.1):
    u, u_t, u_xx = compute_derivatives(model, inputs)
    pde_residual = u_t - alpha * u_xx
    return tf.reduce_mean(tf.square(pde_residual))

def boundary_loss(model, bc_points, L=1.0):
    """Boundary loss for domain [-L, L]"""
    x_bc = bc_points[:, 0:1]
    u_pred = model(bc_points)

    # Left boundary (x = -L): u = 0
    left_mask = tf.cast(tf.abs(x_bc + L) < 1e-5, tf.float32)
    left_loss = tf.reduce_mean(left_mask * tf.square(u_pred))

    # Right boundary (x = +L): u = 0
    right_mask = tf.cast(tf.abs(x_bc - L) < 1e-5, tf.float32)
    right_loss = tf.reduce_mean(right_mask * tf.square(u_pred))

    return left_loss + right_loss

def heat_2d_loss(model, inputs, alpha=0.1):
    """Physics loss for 2D heat equation: u_t = alpha (u_xx + u_yy)."""
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(inputs)
        with tf.GradientTape() as tape1:
            tape1.watch(inputs)
            u = model(inputs)
        grads = tape1.gradient(u, inputs)  # [∂u/∂x, ∂u/∂y, ∂u/∂t]
        u_x = grads[:, 0:1]
        u_y = grads[:, 1:2]
        u_t = grads[:, 2:3]

    u_xx = tape2.gradient(u_x, inputs)
    u_yy = tape2.gradient(u_y, inputs)

    # Check if gradients were computed correctly
    if u_xx is None or u_yy is None:
        raise ValueError("Second-order gradients could not be computed. Check if `inputs` is watched properly.")

    u_xx = u_xx[:, 0:1]
    u_yy = u_yy[:, 1:2]

    del tape2

    pde_res = u_t - alpha * (u_xx + u_yy)
    return tf.reduce_mean(tf.square(pde_res))


def boundary_loss_2d(model, bc_points, L=1.0):
    """Zero Dirichlet BCs on all edges (x=±L or y=±L)."""
    u_pred = model(bc_points)
    x, y = bc_points[:, 0:1], bc_points[:, 1:2]

    # Masks for edges
    mask_x_left  = tf.cast(tf.abs(x + L) < 1e-5, tf.float32)
    mask_x_right = tf.cast(tf.abs(x - L) < 1e-5, tf.float32)
    mask_y_bottom = tf.cast(tf.abs(y + L) < 1e-5, tf.float32)
    mask_y_top    = tf.cast(tf.abs(y - L) < 1e-5, tf.float32)

    loss = (
        tf.reduce_mean(mask_x_left * tf.square(u_pred)) +
        tf.reduce_mean(mask_x_right * tf.square(u_pred)) +
        tf.reduce_mean(mask_y_bottom * tf.square(u_pred)) +
        tf.reduce_mean(mask_y_top * tf.square(u_pred))
    )
    return loss

def heat_3d_loss(model, inputs, alpha=0.1):
    """Physics loss for 3D heat equation: u_t = alpha (u_xx + u_yy + u_zz)."""
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(inputs)
        with tf.GradientTape() as tape1:
            tape1.watch(inputs)
            u = model(inputs)
        grads = tape1.gradient(u, inputs)  # [∂u/∂x, ∂u/∂y, ∂u/∂z, ∂u/∂t]
        u_x, u_y, u_z, u_t = grads[:, 0:1], grads[:, 1:2], grads[:, 2:3], grads[:, 3:4]

    u_xx = tape2.gradient(u_x, inputs)[:, 0:1]
    u_yy = tape2.gradient(u_y, inputs)[:, 1:2]
    u_zz = tape2.gradient(u_z, inputs)[:, 2:3]

    del tape2

    pde_res = u_t - alpha * (u_xx + u_yy + u_zz)
    return tf.reduce_mean(tf.square(pde_res))

def boundary_loss_3d(model, bc_points, L=1.0):
    """Zero Dirichlet BCs on all cube faces."""
    u_pred = model(bc_points)
    x, y, z = bc_points[:, 0:1], bc_points[:, 1:2], bc_points[:, 2:3]

    masks = [
        tf.cast(tf.abs(x + L) < 1e-5, tf.float32),
        tf.cast(tf.abs(x - L) < 1e-5, tf.float32),
        tf.cast(tf.abs(y + L) < 1e-5, tf.float32),
        tf.cast(tf.abs(y - L) < 1e-5, tf.float32),
        tf.cast(tf.abs(z + L) < 1e-5, tf.float32),
        tf.cast(tf.abs(z - L) < 1e-5, tf.float32),
    ]

    loss = sum(tf.reduce_mean(m * tf.square(u_pred)) for m in masks)
    return loss


def initial_condition_loss(model, ic_points, u_true):
    """Initial condition loss."""
    u_pred = model(ic_points)
    return tf.reduce_mean(tf.square(u_pred - u_true))

