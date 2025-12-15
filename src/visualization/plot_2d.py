import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm

def animate_2d_solution(model, L=1.0, T=1.0, num_frames=40, resolution=100):
    """
    Animate 2D heat diffusion as a heatmap for domain [-L, L].
    - model: trained PINN model
    - L: domain limit (square domain [-L, L])
    - T: total time duration
    - num_frames: number of frames in the animation
    - resolution: grid resolution for heatmap
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(-L, L, resolution)
    y = np.linspace(-L, L, resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    # Initial heatmap
    im = ax.imshow(Z, cmap=cm.jet, origin='lower',
                   extent=[-L, L, -L, L], vmin=0, vmax=1, animated=True)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Temperature', fontsize=12)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('2D Heat Diffusion (PINN)')

    # Update function for animation
    def update(frame):
        t_val = T * frame / (num_frames - 1)
        T_arr = np.full_like(X, t_val)
        inputs = np.stack([X, Y, T_arr], axis=-1).reshape(-1, 3)
        u_pred = model.predict(inputs, verbose=0).reshape(resolution, resolution)
        im.set_array(u_pred)
        ax.set_title(f'2D Heat Diffusion at t = {t_val:.2f}s')
        return [im]

    ani = FuncAnimation(fig, update, frames=num_frames, interval=150, blit=True)
    plt.tight_layout()
    plt.show()
    return ani

def plot_static_snapshots(model, L=1.0, T=1.0, times=[0.0, 0.25, 0.5, 0.75, 1.0], resolution=100):
    """
    Plot static heatmaps at specified time snapshots.
    """
    x = np.linspace(-L, L, resolution)
    y = np.linspace(-L, L, resolution)
    X, Y = np.meshgrid(x, y)

    fig, axes = plt.subplots(1, len(times), figsize=(4*len(times), 4))
    for ax, t_val in zip(axes, times):
        T_arr = np.full_like(X, t_val)
        inputs = np.stack([X, Y, T_arr], axis=-1).reshape(-1, 3)
        u_pred = model.predict(inputs, verbose=0).reshape(resolution, resolution)

        im = ax.imshow(u_pred, cmap=cm.jet, origin='lower',
                       extent=[-L, L, -L, L], vmin=0, vmax=1)
        ax.set_title(f't = {t_val:.2f}s')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    fig.colorbar(im, ax=axes.ravel().tolist(), label='Temperature')
    plt.tight_layout()
    plt.show()
