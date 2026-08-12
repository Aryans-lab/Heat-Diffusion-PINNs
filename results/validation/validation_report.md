# PINN Validation Report

Trained models evaluated against the analytic heat-equation solution (sine series with zero Dirichlet BCs). No retraining involved.

## 1D (L=1.0, T=1.0, alpha=0.1, IC amplitude=1.0, width=0.1)

| t | L2 relative error | max abs error |
|---|---|---|
| 0.00 | 3.667e-03 | 2.491e-03 |
| 0.25 | 2.788e-03 | 1.565e-03 |
| 0.50 | 3.857e-03 | 2.344e-03 |
| 0.75 | 2.259e-03 | 8.578e-04 |
| 1.00 | 9.555e-03 | 5.832e-03 |

| Physics residual | RMS value |
|---|---|
| PDE | 2.814e-03 |
| Boundary | 1.972e-03 |
| Initial condition | 1.097e-03 |

Series self-check: max |IC - series(t=0)| = 1.680e-15

## 2D (L=1.0, T=1.0, alpha=0.1, IC amplitude=1.5, width=0.25)

| t | L2 relative error | max abs error |
|---|---|---|
| 0.00 | 2.086e-02 | 1.855e-02 |
| 0.25 | 4.704e-02 | 1.888e-02 |
| 0.50 | 6.865e-02 | 1.952e-02 |
| 0.75 | 8.454e-02 | 2.054e-02 |
| 1.00 | 9.604e-02 | 2.238e-02 |

| Physics residual | RMS value |
|---|---|
| PDE | 1.388e-02 |
| Boundary | 9.911e-03 |
| Initial condition | 6.897e-03 |

Series self-check: max |IC - series(t=0)| = 5.032e-04

## 3D (L=1.0, T=1.0, alpha=0.1, IC amplitude=1.5, width=0.25)

| t | L2 relative error | max abs error |
|---|---|---|
| 0.00 | 2.254e-02 | 2.577e-02 |
| 0.25 | 4.180e-02 | 9.230e-03 |
| 0.50 | 5.874e-02 | 9.297e-03 |
| 0.75 | 7.131e-02 | 1.032e-02 |
| 1.00 | 8.247e-02 | 1.068e-02 |

| Physics residual | RMS value |
|---|---|
| PDE | 4.553e-03 |
| Boundary | 3.764e-03 |
| Initial condition | 3.497e-03 |

Series self-check: max |IC - series(t=0)| = 5.032e-04
