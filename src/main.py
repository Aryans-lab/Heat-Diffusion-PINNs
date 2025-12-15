"""
Main entry point for Physics-Informed Neural Network (PINN) solvers
applied to the heat diffusion equation in 1D, 2D, and 3D.

Author: Aryan Bandyopadhyay
"""

import argparse
from pathlib import Path
import tensorflow as tf

from training.train_1d import train_1d_model
from training.train_2d import train_2d_model
from training.train_3d import train_3d_model

from visualization.plot_1d import animate_1d_solution
from visualization.plot_2d import animate_2d_solution, plot_static_snapshots
from visualization.plot_3d import animate_3d_solution


ROOT = Path(__file__).resolve().parent.parent
SAVED_MODELS = ROOT / "saved_models"
RESULTS = ROOT / "results"


def main():
    parser = argparse.ArgumentParser(
        description="Physics-Informed Heat Diffusion Solver"
    )

    parser.add_argument('--dim', type=int, choices=[1, 2, 3], required=True,
                        help='Dimension: 1 for 1D, 2 for 2D, 3 for 3D')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--visualize', action='store_true', help='Visualize solution')
    parser.add_argument('--static', action='store_true', help='Plot static time snapshots')
    parser.add_argument('--L', type=float, default=1.0, help='Spatial domain length')
    parser.add_argument('--T', type=float, default=1.0, help='Time domain length')

    args = parser.parse_args()

    if not (args.train or args.visualize or args.static):
        raise ValueError("Specify at least one action: --train, --visualize, or --static")

    if args.train and args.visualize:
        print("Warning: Visualization will use saved model, not freshly trained one.")

    if args.dim == 1:
        if args.train:
            print("Training 1D model...")
            train_1d_model()

        if args.visualize:
            model = tf.keras.models.load_model(SAVED_MODELS / "pinn_1d.keras", compile=False)
            ani = animate_1d_solution(model)
            ani.save(RESULTS / "heat_1d.gif")

    elif args.dim == 2:
        if args.train:
            print("Training 2D model...")
            train_2d_model()

        if args.visualize:
            model = tf.keras.models.load_model(SAVED_MODELS / "pinn_2d.keras", compile=False)
            ani = animate_2d_solution(model, L=args.L, T=args.T)
            ani.save(RESULTS / "heat_2d.gif")

        if args.static:
            model = tf.keras.models.load_model(SAVED_MODELS / "pinn_2d.keras", compile=False)
            plot_static_snapshots(model, L=args.L, T=args.T,
                                  times=[0.0, 0.25, 0.5, 0.75, 1.0])

if __name__ == "__main__":
    main()
