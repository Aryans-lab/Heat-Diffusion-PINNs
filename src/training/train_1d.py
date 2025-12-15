import tensorflow as tf
from data.generate_1d import generate_1d_data, gaussian_initial_condition
from models.pinn import build_pinn
from training.losses import heat_1d_loss, boundary_loss, initial_condition_loss

def train_1d_model():
    """Train 1D heat equation solver"""
    # Configuration
    L = 1.0
    T = 1.0
    ALPHA = 0.1
    N_POINTS = 30000
    EPOCHS = 10000
    BATCH_SIZE_COLL = 2000
    BATCH_SIZE_BC = 2000
    BATCH_SIZE_IC = 2000

    # Generate data
    data = generate_1d_data(N_POINTS, L, T)
    ic_func = lambda x: gaussian_initial_condition(x)

    # Create TensorFlow datasets with batching
    coll_ds = tf.data.Dataset.from_tensor_slices(data['collocation']).batch(BATCH_SIZE_COLL)
    bc_ds = tf.data.Dataset.from_tensor_slices(data['boundary']).batch(BATCH_SIZE_BC)
    ic_ds = tf.data.Dataset.from_tensor_slices(data['initial']).batch(BATCH_SIZE_IC)

    # Build model
    model = build_pinn(input_dim=2, hidden_units=60, num_layers=4)
    optimizer = tf.keras.optimizers.Adam(1e-4)

    @tf.function
    def train_step(coll_batch, bc_batch, ic_batch):
        with tf.GradientTape() as tape:
            pde_loss_val = heat_1d_loss(model, coll_batch, ALPHA)
            bc_loss_val = boundary_loss(model, bc_batch, L=L)
            ic_loss_val = initial_condition_loss(model, ic_batch, ic_func(ic_batch[:, 0:1]))

            # Equal weights
            total_loss = pde_loss_val + bc_loss_val + ic_loss_val

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss, pde_loss_val, bc_loss_val, ic_loss_val

    # Training loop
    for epoch in range(EPOCHS):
        for coll_batch, bc_batch, ic_batch in zip(coll_ds, bc_ds, ic_ds):
            total_loss, pde_loss, bc_loss, ic_loss = train_step(coll_batch, bc_batch, ic_batch)

        # Print diagnostics every 200 epochs
        if epoch % 200 == 0:
            print(f"Epoch {epoch}:")
            print(f"  Total Loss = {total_loss:.4e}")
            print(f"  PDE loss   = {pde_loss:.4e}")
            print(f"  BC loss    = {bc_loss:.4e}")
            print(f"  IC loss    = {ic_loss:.4e}")

    # Save model
    model.save('saved_models/pinn_1d.keras')
    return model
