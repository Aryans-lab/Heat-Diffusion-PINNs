import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)


def _sample_slice(model, L, z_val, t_val, resolution):
    """Evaluate the PINN on the 2D slice {z = z_val} at time t_val."""
    x = np.linspace(-L, L, resolution)
    y = np.linspace(-L, L, resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.full_like(X, z_val)
    T = np.full_like(X, t_val)
    inputs = np.stack([X, Y, Z, T], axis=-1).reshape(-1, 4)
    u = model.predict(inputs, verbose=0).reshape(resolution, resolution)
    return X, Y, u


def animate_3d_solution(model, L=1.0, T=1.0, num_frames=40, resolution=60):
    """
    Animate 3D heat diffusion as a surface plot of temperature on the
    mid-plane z = 0, evolving over time.
    """
    fig = plt.figure(figsize=(9, 7), layout='constrained')
    ax = fig.add_subplot(111, projection='3d')

    _, _, u0 = _sample_slice(model, L, 0.0, 0.0, resolution)
    zmax = float(u0.max() * 1.05)

    def draw(t_val):
        X, Y, u = _sample_slice(model, L, 0.0, t_val, resolution)
        ax.clear()
        ax.plot_surface(X, Y, u, cmap=cm.jet, linewidth=0,
                        antialiased=True, rstride=1, cstride=1)
        ax.set_xlim(-L, L)
        ax.set_ylim(-L, L)
        ax.set_zlim(0, zmax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('Temperature')
        ax.set_title(f'3D Heat Diffusion at t = {t_val:.2f}s')
        ax.view_init(elev=30, azim=45)

    draw(0.0)

    mappable = cm.ScalarMappable(cmap=cm.jet, norm=plt.Normalize(0, zmax))
    cbar = fig.colorbar(mappable, ax=ax, pad=0.1)
    cbar.set_label('Temperature')

    def update(frame):
        t_val = T * frame / (num_frames - 1)
        draw(t_val)

    ani = FuncAnimation(fig, update, frames=num_frames, interval=150)
    return ani


def plot_static_snapshots_3d(model, L=1.0, T=1.0,
                             times=(0.0, 0.25, 0.5, 0.75, 1.0),
                             z_levels=None, resolution=60):
    """
    Plot heatmaps of the temperature field on slices z = const at several
    times. Rows are z levels, columns are time snapshots.
    """
    if z_levels is None:
        z_levels = (-L / 2, 0.0, L / 2)

    slices = {(z, t): _sample_slice(model, L, z, t, resolution)[2]
              for z in z_levels for t in times}
    vmin = min(u.min() for u in slices.values())
    vmax = max(u.max() for u in slices.values())

    fig, axes = plt.subplots(len(z_levels), len(times),
                             figsize=(3 * len(times), 3 * len(z_levels)),
                             squeeze=False, layout='constrained')

    for i, z in enumerate(z_levels):
        for j, t in enumerate(times):
            ax = axes[i, j]
            u = slices[(z, t)]
            im = ax.imshow(u, cmap=cm.jet, origin='lower',
                           extent=[-L, L, -L, L], vmin=vmin, vmax=vmax)
            ax.set_title(f'z = {z:.2f}, t = {t:.2f}s' if i == 0 else f't = {t:.2f}s')
            ax.set_xlabel('x' if i == len(z_levels) - 1 else '')
            ax.set_ylabel('y' if j == 0 else '')

    fig.colorbar(im, ax=axes.ravel().tolist(), label='Temperature')
    return fig
