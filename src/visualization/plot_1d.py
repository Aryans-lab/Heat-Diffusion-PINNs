import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_1d_solution(model, L=1.0, T=1.0, num_frames=50):
    """Create animation of 1D heat diffusion"""
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(-L, L, 200)  # ✅ Domain updated to [-L, L]
    line, = ax.plot([], [], 'b-', lw=2)
    
    def init():
        ax.set_xlim(-L, L)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel('Position (x)')
        ax.set_ylabel('Temperature (u)')
        ax.grid(True)
        return line,
    
    def update(frame):
        t_val = T * frame / num_frames
        t = np.full_like(x, t_val)
        inputs = np.stack([x, t], axis=1)
        u_pred = model.predict(inputs, verbose=0)
        line.set_data(x, u_pred)
        ax.set_title(f'Heat Diffusion at t = {t_val:.2f}s')
        return line,
    
    ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True)
    plt.close()
    return ani
