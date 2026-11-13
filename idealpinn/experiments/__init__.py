"""Experiments.  Each module exposes PRESETS, run(cfg, out) and plot(out).

Exact-theory experiments (NumPy only, minutes on a laptop CPU):
    bias_vs_weight, misspecification, uv_divergence
Neural experiments (PyTorch, GPU recommended for the "paper" preset):
    pinn_vs_ideal, pinn_ensemble, adaptive_weights, forward_benchmark
"""

REGISTRY = ["bias_vs_weight", "misspecification", "uv_divergence",
            "pinn_vs_ideal", "pinn_ensemble", "adaptive_weights", "forward_benchmark"]
