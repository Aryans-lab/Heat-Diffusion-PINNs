# Theory notes

These notes define every quantity computed in `idealpinn.exact` and state what
is derived, what is checked numerically, and what is still open. They are
working notes for the paper, not the paper.

## 1. Setting

Heat equation on the unit box Ω = [0,1]^d (d = 1, 2, 3), 0 ≤ t ≤ T:

    u_t = α ∇²u,    u = 0 on ∂Ω,    u(x, 0) = u0(x).

u0 is a Gaussian bump projected on the first `ic_modes` sine modes per
direction, so the exact solution is a finite sum

    m_α(x, t) = Σ_k c_k φ_k(x) exp(−α μ_k t),
    φ_k(x) = Π_j √2 sin(k_j π x_j),    μ_k = π² |k|².

Data: N noisy point readings y_i = u(z_i) + ε_i, z_i = (x_i, t_i),
ε_i ~ N(0, σ²). The unknown is the diffusivity α; the true value is α0 = 0.1.

## 2. The PINN objective as a statistical model

An inverse PINN with exact initial and boundary conditions minimises, over the
network u_θ and over α,

    L = w_data · (1/N) Σ_i (u(z_i) − y_i)² + w_res · (1/N_c) Σ_c (u_t − α∇²u)²(z_c).

The collocation mean is a Monte-Carlo estimate of (1/|Ω|T) ∫∫ (u_t − α∇²u)².
Multiplying L by N/(w_data σ²) gives

    S[u, α] = Σ_i (u(z_i) − y_i)²/σ² + β ∫∫ (u_t − α∇²u)² dx dt,
    β = (w_res / w_data) · N / (σ² |Ω| T).                                   (1)

exp(−S/2) is, up to normalisation, the joint density of the data and of a field
obeying the *stochastic* heat equation

    u_t = α∇²u + ξ,    ⟨ξ(x,t) ξ(x',t')⟩ = β⁻¹ δ(x−x') δ(t−t').             (2)

So the PDE-residual penalty is the Onsager–Machlup action of (2): a soft
physics loss asserts that the PDE holds only up to white-noise forcing of
intensity 1/β. This identification is known (physics-informed information
field theory, Alberts & Bilionis 2023; kernel view of physics-informed
learning, Doumèche et al. 2024). What this project adds is carrying it through
to the *parameter estimator* in closed form and testing it against trained
networks.

Under (2), u = m_α + v with v a centred Gaussian field of covariance C_α/β,

    C_α(z, z') = Σ_k φ_k(x) φ_k(x') · [exp(−αμ_k |t−t'|) − exp(−αμ_k (t+t'))] / (2 α μ_k).   (3)

`HeatBox.kernel` evaluates (3) with a spectral cut-off (`n_modes`).
`tests/test_exact.py::test_kernel_matches_simulated_stochastic_heat_equation`
checks it against a direct simulation of the mode amplitudes.

## 3. Four estimators from one action

Write r = y − m_α (residual of the hard-constrained solution at the sensors)
and K_α = σ² I + C_α/β (N × N).

**Tree level (the ideal PINN).** Minimising S over the field at fixed α is a
linear-quadratic problem with solution

    û = m_α + C_α(·, data) (σ²β I + C_α)⁻¹ r,
    J_tree(α) = min_u S[u, α] = rᵀ K_α⁻¹ r.                                   (4)

An inverse PINN that is expressive enough and trained to the global minimum
returns α̂_tree = argmin J_tree. `test_tree_level_equals_bruteforce_minimisation_over_the_field`
verifies (4) against a direct discretisation of the field, with no kernel
involved.

**One loop (marginal likelihood).** Integrating the field out instead of
minimising over it gives the exact marginal likelihood of model (2):

    J_1loop(α) = −2 log p(y | α, β) = rᵀ K_α⁻¹ r + log det(I + C_α/(βσ²)) + const.   (5)

The extra term is the Gaussian fluctuation determinant. It is α-dependent
because C_α is. The PINN objective omits it.

**Hard constraint (MLE).** β → ∞ gives J_mle = |r|²/σ², ordinary non-linear
least squares on the exact solution; this is efficient when the PDE is exact.

**Profiled.** J_prof(α) = Σ_i (û(z_i) − y_i)²/σ² = σ² rᵀ K_α⁻² r: the data misfit
of the soft-constrained field, i.e. the outer criterion of bilevel /
generalised-profiling schemes if their inner problem is a soft PINN fit.

**Learned β.** Minimise (5) jointly over α and β (β = ∞ allowed).

### Why tree level is biased

A smaller diffusivity damps fluctuations more slowly, so the same forcing
produces a larger model error: ∂C_α/∂α is negative definite on the sensor set
(checked numerically for the default design: all 100 eigenvalues are negative;
not proved in general). Lowering α therefore makes K_α larger and rᵀK_α⁻¹r
smaller for *any* residual. The PINN objective rewards this; the
log-determinant in (5) is the term that charges for it. Hence α̂_tree is pulled
low and, when β is fixed at a value that is too small, α̂_1loop is pushed high.

### A dimensionless control parameter

    ρ = (prior std of the model error at the sensors) / σ = √(mean_i C_α0(z_i,z_i) / β) / σ.

From (1), w_res/w_data = s² |Ω|T / (ρ² N), with s² = mean_i C_α0(z_i, z_i).
ρ ≪ 1: physics trusted, all estimators → MLE. ρ ≳ 1: the fit is allowed to
bend the field by more than the noise.

For the default design (5 sensors × 20 times, σ = 0.02) equal loss weights
correspond to ρ ≈ 0.084.

## 4. What is measured

All figures of merit are frequentist, over repeated pseudo-experiments, in
units of the Cramér–Rao bound of the exact-physics model,
CRB = σ / ‖∂m_α/∂α‖:

* bias/CRB, standard deviation/CRB, RMSE/CRB;
* coverage of the nominal 68 % interval {α : J(α) ≤ min J + 1}.

Three data-generating worlds (`idealpinn.exact.study.World`):

* `exact`      — the PDE holds exactly;
* `stochastic` — data generated from (2) with ratio ρ_true (the soft-physics
                 model is *true*);
* `loss`       — deterministic unmodelled cooling, u_t = α∇²u − h u.

## 5. Dependence on the resolution (d ≥ 2)

The point variance of (3), s²(Λ) = Σ_{|k|≤Λ} φ_k(x)² (1 − e^{−2αμ_k t})/(2αμ_k),
behaves like Σ_k 1/|k|²: finite in d = 1, logarithmically divergent in d = 2,
linearly divergent in d = 3. With *point* measurements the infimum of S over
all fields is then not attained in the continuum for d ≥ 2: the field can match
each reading with an arbitrarily narrow, arbitrarily cheap spike (point
evaluation is not continuous on the space where S is finite; the stochastic
heat equation has no function-valued solution there). In practice the
network's finite capacity and the collocation density act as the cut-off Λ.
The experiment `uv_divergence` shows how the ideal estimator at fixed loss
weights moves with Λ: slowly (logarithmically) in d = 2.

## 6. What is and is not claimed

Derived, and verified numerically by independent computations in `tests/`:

1. (3)–(5) and the conversion (1) between loss weights and β.

Observed with trained networks, on smoke runs only:

2. A small network trained with Levenberg–Marquardt reaches α̂_tree to within
   0.1 CRB at ρ = 0.1, 0.3 and 1 on one data set (`pinn_vs_ideal`, quick
   preset). The ensemble version of this test (`pinn_ensemble`, paper preset)
   has not been run.

Empirical findings from the experiments (see `results/`):

3. With equal weights the ideal PINN is practically the MLE for the designs
   studied. Bias appears once ρ ≳ 0.3 and grows with N at fixed ρ.
4. When the model error is real (stochastic world), one loop with the right or
   a learned β is close to the Cramér–Rao bound of that world and calibrated;
   the MLE interval is strongly over-confident; tree level is biased.
5. No estimator here removes a *deterministic* missing-physics bias; apparent
   successes at particular ρ are cancellations.

Not claimed / open:

* "Inverse PINNs are biased" is not new on its own (e.g. Atchadé & Mukherjee
  2026 for plug-in estimators; empirical reports in the benchmark literature).
  The contribution is the closed-form characterisation, the sign and size of
  the bias in terms of ρ, the identification of the missing determinant, and
  the network-level check.
* Everything is for a linear PDE with a parameter entering linearly in the
  operator. For non-linear PDEs (5) becomes a Laplace approximation.
* Networks used for the check are small (≈ 1–2 k parameters) and the check is
  in d = 1.
* Statements about adaptive weighting concern the schemes implemented here
  (gradient-norm balancing), under the settings in the presets.

## 7. References

* M. Raissi, P. Perdikaris, G. E. Karniadakis, J. Comput. Phys. 378 (2019) 686.
* A. Alberts, I. Bilionis, "Physics-informed information field theory for
  modeling physical systems with uncertainty quantification", J. Comput. Phys.
  486 (2023) 112100, arXiv:2301.07609.
* A. Alberts, I. Bilionis, arXiv:2401.14224.
* N. Doumèche, F. Bach, G. Biau, C. Boyer, "Physics-informed machine learning
  as a kernel method", COLT 2024; "Physics-informed kernel learning",
  arXiv:2409.13786.
* M. Dashti, K. J. H. Law, A. M. Stuart, J. Voss, "MAP estimators and their
  consistency in Bayesian nonparametric inverse problems", Inverse Problems 29
  (2013) 095017.
* R. Z. Zhang, X. Xie, J. Lowengrub, "BiLO: Bilevel local operator learning
  for PDE inverse problems", arXiv:2404.17789.
* S. Wang, Y. Teng, P. Perdikaris, SIAM J. Sci. Comput. 43 (2021) A3055
  (gradient-norm balancing).
* S. Wang, S. Sankaran, P. Perdikaris, arXiv:2203.07404 (causal training).
* N. McGreivy, A. Hakim, Nature Machine Intelligence 6 (2024) 1256.
* Y. Atchadé, S. Mukherjee, arXiv:2609.12301.
* J. O. Ramsay, G. Hooker, D. Campbell, J. Cao, J. R. Stat. Soc. B 69 (2007)
  741 (generalised profiling).

Check every reference against the original before citing it in the paper;
several were taken from search summaries, not from the full text.
