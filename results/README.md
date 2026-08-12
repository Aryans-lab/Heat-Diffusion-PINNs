## Results

This directory contains visualization outputs generated from trained PINNs,
including static plots and animated heat diffusion results for 1D, 2D, and 3D cases.

### `validation/`

Quantitative validation of the trained models against the analytic
heat-equation solution (sine series with zero Dirichlet boundary conditions):

- `validation_report.md` — L2 relative / max-absolute errors over time and
  PDE / boundary / initial-condition residuals for each dimension.
- `error_summary.csv` — machine-readable version of the error tables.
- `1d_error.png`, `2d_error.png`, `3d_error.png` — pointwise error maps.

Regenerate with: `python -m validation.evaluate` (from `src/`). No retraining
is required; the script loads the saved models from `saved_models/`.
