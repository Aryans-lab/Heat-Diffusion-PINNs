# Heat Diffusion with Physics-Informed Neural Networks (PINNs)

> Solving the heat equation **∂u/∂t = α ∇²u** in **1D, 2D, and 3D** with
> mesh-free physics-informed neural networks — no labeled data, just the
> governing PDE baked into the loss.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Table of Contents

- [Overview](#overview)
- [The Physics](#the-physics)
- [The Method](#the-method)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Validation Against the Exact Solution](#validation-against-the-exact-solution)
- [Key Findings](#key-findings)
- [Limitations & Future Work](#limitations--future-work)
- [Author](#author)

---

## Overview

Partial differential equations (PDEs) govern everything from thermal conduction
to diffusion in fluids and materials. Traditional solvers (finite difference /
finite element) rely on structured grids and become expensive in higher
dimensions or on complex domains.

**Physics-Informed Neural Networks (PINNs)** take a different route: a neural
network represents the solution field `u(x, t)`, and the *physics is embedded
directly in the loss function* through automatic differentiation. This makes the
approach **mesh-free** and dimension-agnostic — the same code trains a 1D, 2D,
or 3D solver.

This project applies PINNs to the heat (diffusion) equation in 1D, 2D, and 3D,
and — crucially — **validates the trained models against the analytic solution**,
so the results are measured, not just visualized.

---

## The Physics

### Governing equation

On a hypercube domain `[-L, L]^d` and time interval `[0, T]`:

\[
\frac{\partial u}{\partial t} \;=\; \alpha\,\nabla^2 u
\]

with

- `u(x, t)` — temperature field,
- `α` — thermal diffusivity (`α = 0.1` here),
- `∇²` — Laplacian in the spatial coordinates.

### Initial condition

A Gaussian "temperature bump" centered at the origin:

\[
u(\mathbf{x}, 0) \;=\; A\,\exp\!\left(-\frac{|\mathbf{x}|^2}{2\sigma^2}\right)
\]

### Boundary conditions

Zero Dirichlet conditions on every face of the domain:

\[
u(\mathbf{x}, t)\big|_{\partial[-L,L]^d} = 0
\]

Physically: heat diffuses out of the Gaussian bump, spreads toward the walls,
and the solution decays to zero everywhere.

---

## The Method

A fully connected `tanh` network `u_θ(x, t)` approximates the solution. Instead
of training on data, it is trained to **satisfy the physics** by minimizing a
composite loss:

\[
\mathcal{L} \;=\; \underbrace{\mathcal{L}_{\text{PDE}}}_{\text{residual}}
        \;+\; \underbrace{\mathcal{L}_{\text{IC}}}_{\text{initial condition}}
        \;+\; \underbrace{\mathcal{L}_{\text{BC}}}_{\text{boundary condition}}
\]

| Term | Formula | What it enforces |
|---|---|---|
| PDE residual | `‖ ∂u_θ/∂t − α ∇²u_θ ‖²` | the heat equation holds at collocation points |
| IC loss | `‖ u_θ(x, 0) − u_IC(x) ‖²` | matches the Gaussian bump at `t = 0` |
| BC loss | `‖ u_θ ‖²` on the boundary | temperature is pinned to zero at the walls |

The spatial and temporal derivatives `∂u/∂t`, `∂u/∂x`, `∂²u/∂x²`, ... are
computed with **automatic differentiation** (`tf.GradientTape`), so no
discretization or meshing is ever needed. Training uses random collocation
points sampled uniformly in space–time, and Adam to minimize the composite loss.

---

## Project Structure

```
.
├── README.md
├── conftest.py                 # pytest path setup for the src/ package
├── data/                       # no raw data needed (points are generated on the fly)
├── results/
│   ├── figures/                # animations & snapshots (heat_1d.gif, ...)
│   └── validation/             # error reports, CSVs and error maps
├── saved_models/               # trained pinn_1d/2d/3d.keras
├── src/
│   ├── main.py                 # CLI entry point (train / visualize / validate)
│   ├── data_generation/        # collocation, boundary & IC point samplers
│   ├── models/                 # build_pinn network factory
│   ├── training/               # losses.py + the three training scripts
│   ├── validation/             # analytic solution & error evaluation
│   └── visualization/          # 1D/2D/3D animations & static plots
└── tests/                      # finite-difference checks on the derivative code
```

---

## Installation

```bash
# Clone & install (Python 3.11+, TensorFlow 2.x)
git clone https://github.com/Aryans-lab/Heat-Diffusion-PINNs.git
cd Heat-Diffusion-PINNs
pip install -r requirements.txt   # or: pip install tensorflow numpy matplotlib
```

---

## Usage

Everything is driven from `src/main.py`. Run from inside `src/`:

```bash
# Train a 1D model (also 2D / 3D with --dim 2 / --dim 3)
python main.py --dim 1 --train

# Visualize with the saved model (saves results/figures/heat_1d.gif)
python main.py --dim 1 --visualize

# Static time snapshots (2D / 3D)
python main.py --dim 2 --static

# Validate the saved models against the exact solution
python -m validation.evaluate

# Run the derivative tests
python -m pytest tests/
```

| Flag | Effect |
|---|---|
| `--dim {1,2,3}` | space dimension to solve |
| `--train` | train a fresh model |
| `--visualize` | animate the solution with the saved model |
| `--static` | static time snapshots (2D/3D) |
| `--L`, `--T` | domain half-length and time horizon |

---

## Results

### Qualitative behavior

- A smooth Gaussian bump that diffuses and flattens over time.
- Radial / volumetric spreading in 2D and 3D that matches the physics.
- Stable, physically meaningful predictions without any grid-based solver.

### Quantitative validation (measured, not just plotted)

Each saved model was evaluated against the **analytic heat-equation solution**
(separation of variables → sine series). `L2` error is relative to the
norm of the exact field.

| Dimension | L2 rel. error @ t = 0 | L2 rel. error @ t = 1 | PDE residual (RMS) |
|---|---:|---:|---:|
| **1D** | 3.7e-3 | 9.6e-3 | 2.8e-3 |
| **2D** | 2.1e-2 | 9.6e-2 | 1.4e-2 |
| **3D** | 2.3e-2 | 8.2e-2 | 4.6e-3 |

The 1D solver is accurate to ~1%; higher-dimensional cases degrade to ~10% at
long times — a typical signature of a *plain* PINN (no loss reweighting or
causal training), and a clear target for the improvements listed below.

Full tables, residuals and error maps live in
[`results/validation/`](results/validation/validation_report.md).

---

## Validation Against the Exact Solution

Because the heat equation on `[-L, L]^d` with zero Dirichlet conditions has a
known analytic solution, this project measures the models instead of trusting
them:

**Separation of variables.** With `k_n = nπ/(2L)` and a separable Gaussian
initial condition `f(x₁, …, x_d) = ∏ᵢ g(xᵢ)`,

\[
u(\mathbf{x}, t) = \prod_{i=1}^{d} \sum_{n\ge 1}
B_n\, e^{-\alpha k_n^2 t}\,\sin\!\big(k_n (x_i + L)\big),
\qquad
B_n = \frac{1}{L}\int_{-L}^{L} g(x)\sin\!\big(k_n(x+L)\big)\,dx .
\]

`src/validation/analytic.py` builds this series, and
`src/validation/evaluate.py` reports:

- **L2 relative & max-absolute errors** over time,
- **PDE / boundary / IC residuals** computed by the very loss functions used in
  training (a direct measure of how well the physics is satisfied),
- **pointwise error maps** (`results/validation/*d_error.png`).

A self-check verifies the series reproduces the initial condition to machine
precision before any error is reported.

The derivative code — the heart of any PINN — is guarded by
**finite-difference tests** (`tests/test_derivatives.py`): the analytical
first/second derivatives from `GradientTape` are compared against central
finite differences, and the generic derivative helper is checked to reproduce
each physics loss exactly.

---

## Key Findings

- PINNs solve diffusion-type PDEs **without any labeled data**.
- Random collocation sampling avoids mesh generation entirely.
- Higher-dimensional problems reuse the same architecture and training loop.
- The saved models are physically correct at short times; accuracy degrades
  over long horizons for 2D/3D — motivating modern PINN techniques.

---

## Limitations & Future Work

| Limitation | Planned fix |
|---|---|
| Accuracy degrades at late times (esp. 2D/3D) | Causal training & NTK-based / self-adaptive loss weighting |
| Equal loss weighting is fragile | Hard-constrained IC/BC via output transforms |
| Sharp gradients not resolved | Residual-based adaptive refinement (RAR), Fourier features |
| Slow convergence | L-BFGS refinement after Adam, quasi-Monte-Carlo sampling |
| Fixed diffusivity | Inverse problems: *identify* `α(x)` from noisy data |
| Linear PDE only | Nonlinear equations (Burgers, Allen–Cahn) |

---

## Author

**Aryan Bandyopadhyay** · Integrated MSc Physics
School of Physical Sciences, NISER Bhubaneswar
