import tensorflow as tf
from data.generate_2d import generate_2d_square, gaussian_initial_condition_2d
from models.pinn import build_pinn
from training.losses import heat_2d_loss, boundary_loss_2d, initial_condition_loss

def train_2d_model():
    """Train 2D heat equation PINN for domain [-L, L]."""
    L, T, ALPHA = 1.0, 1.0, 0.1
    N_POINTS, EPOCHS, BATCH_SIZE = 60000, 10000, 2000

    # Generate data
    data = generate_2d_square(N_POINTS, L, T)
    ic_func = lambda x, y: gaussian_initial_condition_2d(x, y)

    # Build model
    model = build_pinn(input_dim=3, hidden_units=64, num_layers=6)
    optimizer = tf.keras.optimizers.Adam(5e-4)

    # Convert data to batched datasets
    def to_dataset(arr): 
        return tf.data.Dataset.from_tensor_slices(arr).shuffle(10000).batch(BATCH_SIZE).repeat()

    coll_ds = to_dataset(data['collocation'])
    bc_ds = to_dataset(data['boundary'])

    # Generate target values for initial condition
    ic_vals = ic_func(data['initial'][:, 0:1], data['initial'][:, 1:2])

    # Use zip to ensure each (input, label) pair is preserved
    ic_dataset = tf.data.Dataset.from_tensor_slices((data['initial'], ic_vals)) \
                            .shuffle(10000) \
                            .batch(BATCH_SIZE) \
                            .repeat()

    # Create iterators
    coll_iter = iter(coll_ds)
    bc_iter = iter(bc_ds)
    ic_iter = iter(ic_dataset)


    # Training step
    @tf.function
    def train_step(coll_batch, bc_batch, ic_batch, ic_true_batch):
        with tf.GradientTape() as tape:
            pde_loss = heat_2d_loss(model, coll_batch, ALPHA)
            bc_loss_val = boundary_loss_2d(model, bc_batch, L)
            ic_loss_val = initial_condition_loss(model, ic_batch, ic_true_batch)
            total_loss = pde_loss + bc_loss_val + ic_loss_val

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss, pde_loss, bc_loss_val, ic_loss_val

    # Training loop
    for epoch in range(EPOCHS):
        coll_b = next(coll_iter)
        bc_b = next(bc_iter)
        ic_b, ic_val_b = next(ic_iter)  # <-- updated line

        total_loss, pde_loss, bc_loss, ic_loss = train_step(coll_b, bc_b, ic_b, ic_val_b)

        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Total={total_loss:.4e} | PDE={pde_loss:.4e} | BC={bc_loss:.4e} | IC={ic_loss:.4e}")

    model.save('saved_models/pinn_2d.keras')
    return model

