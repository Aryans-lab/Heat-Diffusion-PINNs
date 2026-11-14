# idealpinn — what does an inverse PINN actually estimate?

A physics-informed neural network that infers a PDE coefficient from noisy data
minimises *data misfit + weight × PDE residual*. This repository answers, for
the heat equation, a question that is usually left to trial and error:

> If the network were perfectly expressive and perfectly trained, which number
> would it return, how far is that number from the best possible estimate, and
> what does the loss weight have to do with it?

The answer is computed in closed form (no network involved) and then checked
against real PINNs trained on the same data sets.

**Status: research code with first results, not a finished paper.** What has
and has not been run is listed under [Status](#status). Read
[Limitations](#limitations-and-open-questions) before quoting anything.

## The idea in five lines

* The L² residual penalty is the action of the *stochastic* heat equation
  u_t = α∇²u + ξ. The loss-weight ratio fixes the noise strength 1/β of ξ.
* Minimising the PINN loss over the field can be done exactly. What remains is
  a function of α alone: `tree(α) = rᵀ(σ²I + C_α/β)⁻¹r`. Its minimiser is the
  **ideal PINN estimate**.
* The exact marginal likelihood of the same model has one more term,
  `log det(I + C_α/(βσ²))`. The PINN objective leaves it out. That omission is
  the source of the bias.
* One dimensionless number controls everything: ρ = (model error allowed by
  the loss) / (measurement noise).
* All of this is compared with the Cramér–Rao bound, over thousands of
  pseudo-experiments, in three worlds: exact physics, random model error,
  unmodelled cooling.

Definitions and derivations: [`docs/THEORY.md`](docs/THEORY.md).

## Results so far

Units: CRB = Cramér–Rao bound of the exact-physics model. Design unless
stated: 1D, 5 sensors × 20 times (N = 100), σ = 0.02, α0 = 0.1.

### 1. Closed-form theory (paper presets, 4000 pseudo-experiments each)

Exact physics, `results/bias_vs_weight/paper/`:

| ρ | ideal PINN bias | ideal PINN sd | coverage of nominal 68 % interval |
|---:|---:|---:|---:|
| 0.03 – 0.1 | −0.00 … −0.01 | 1.01 – 1.02 | 0.68 – 0.73 |
| 0.32 | −0.11 | 1.27 | 0.84 |
| 1.0 | −1.14 | 2.06 | 0.84 |
| 2.2 | −3.97 (3 % of estimates at the scan edge) | 2.47 | 0.78 |

* Equal loss weights correspond to ρ ≈ 0.08 for this design: there the ideal
  PINN is the maximum-likelihood estimate to within the Monte-Carlo error.
* Up-weighting the data moves ρ up. By ρ = 1 the estimate is biased by about
  one CRB and its spread has doubled.
* At fixed ρ = 1 the bias is larger for more data: −0.31 (N = 30), −1.14
  (N = 100), −4.78 (N = 360, 5 % of estimates at the edge of the scan range).
  These three designs also differ in sensor count, so this is not a clean
  scaling law.
* One loop at the same fixed ρ is biased the other way (+0.89 at ρ = 1). With
  ρ learned from the data it is close to the bound again (bias −0.01, sd 1.12).

Imperfect physics, `results/misspecification/paper/table.md` (RMSE/CRB, and
coverage in brackets):

| world | MLE | ideal PINN, ρ = 1 | one loop, ρ = 1 | one loop, learned ρ |
|---|---:|---:|---:|---:|
| exact | 0.98 (0.68) | 2.30 (0.85) | 2.22 (0.88) | 1.05 (0.72) |
| random model error, ρ_true = 1 | 5.72 (0.14) | 3.78 (0.60) | 3.40 (0.69) | 3.44 (0.67) |
| random model error, ρ_true = 3 | 17.8 (0.04) | 10.8 (0.19) | 9.50 (0.23) | 7.05 (0.69) |
| unmodelled cooling, h = 0.05 | 2.74 (0.07) | 2.03 (0.91) | 2.82 (0.77) | 2.63 (0.21) |

* When the model error is real, the hard-constrained fit is badly
  over-confident (14 % coverage instead of 68 %). The learned-ρ marginal
  likelihood stays calibrated and reaches the bound of that world (3.38 CRB
  at ρ_true = 1).
* Nothing here repairs a deterministic missing term. The ideal PINN's good
  score for cooling at ρ = 1 is a cancellation of two biases of opposite sign
  (its bias is +1.85 at ρ = 0.3 and −4.83 at ρ = 3).

Resolution dependence, `results/uv_divergence/paper/`: the point variance of
the implied model-error field converges in 1D (0.806), grows logarithmically
in 2D and linearly in 3D with the spectral cut-off Λ. In 2D at fixed loss
weights the ideal PINN bias moves from −0.64 (Λ = 4) to −2.12 (Λ = 64), with
the steps shrinking (−2.02 at Λ = 32). So with point data the meaning of a
loss weight depends on the resolution of the network when d ≥ 2.

### 2. Trained networks versus the closed form (smoke runs only)

Single data set, tanh MLP with 1298 parameters, 2048 collocation points,
Levenberg–Marquardt, 200 iterations, about 3 minutes on 2 CPU cores
(`results/pinn_vs_ideal/quick/`):

| ρ | (PINN − ideal PINN)/CRB | (PINN − MLE)/CRB | predicted (ideal − MLE)/CRB |
|---:|---:|---:|---:|
| 0.1 | −0.08 | −0.14 | −0.06 |
| 0.3 | −0.01 | −0.40 | −0.39 |
| 1.0 | +0.08 | −2.07 | −2.15 |

On this data set the network lands on the predicted estimate, not on the MLE.
This is one data set and one initialisation.

Because the closed form also gives the value the loss must reach, training can
be checked against it. In the 8-network smoke ensemble
(`results/pinn_ensemble/quick/`, 922 parameters, 1024 collocation points, 250
iterations) the two go together:

| final loss / ideal loss | (PINN − ideal PINN)/CRB |
|---:|---:|
| 1.005, 1.005, 1.007 | −0.10, +0.03, +0.13 |
| 0.989 (below the continuum optimum) | +1.31 |
| 1.07, 1.12, 1.13, 1.36 (not converged; all four are the ρ = 1 networks) | −0.82, −0.17, −0.14, −3.10 |

The experiment labels a network "certified" when its loss is within 1 % of the
ideal value. That threshold was chosen after looking at these eight networks,
so it proves nothing here, and the single-data-set run above shows it is not a
necessary condition (there the ρ = 1 network is at 1.17 and still within
0.08 CRB).
Eight networks are an illustration, not a statistic. The `paper` preset (192
networks) is what tests this properly, and it has not been run.

### 3. Things that went wrong and are part of the story

* **α collapses towards zero** if it is trainable from the first iteration. A
  fresh network has u_t ≈ 0, so the residual is proportional to α and the
  optimiser shrinks α. In the first smoke ensemble 2 of 12 networks ended at
  α = 0.046 and α = 0.000. With α frozen for the first 40 iterations (now the
  default) none of the 8 networks in the rerun collapsed. The two runs differ
  in ensemble size and iteration count as well, so this is not a controlled
  comparison.
* **Adam is slow.** After 3000 steps it is more than 30 CRB from the ideal
  estimate. After 20 000 steps (10 minutes; `results/pinn_vs_ideal/extra_adam20k/`)
  it is still 2.6 to 9.1 CRB short and α is still rising. Levenberg–Marquardt
  is within 0.1 CRB after 3 minutes on the same data. Whether Adam gets there
  in the 60 000–80 000 steps of the `paper` presets is not known.
* **Gradient-norm loss balancing moved the weights to ρ ≈ 0.87** in the smoke
  run (two data sets, 3000 Adam steps), i.e. from the unbiased regime into the
  biased one. The networks were unconverged, so this is only a pointer to what
  `adaptive_weights` should measure.

### 4. Forward problem, 1D smoke run (`results/forward_benchmark/quick/`)

| recipe | relative L² error | time for 2 seeds (2 CPU cores) |
|---|---:|---:|
| legacy: soft constraints, Adam, 3000 steps | 3.3e-2 | 76 s |
| modern: exact constraints, Fourier features, causal weights, Adam | 1.4e-3 | 67 s |
| exact constraints + Levenberg–Marquardt, 150 iterations | 6.0e-4 | 64 s |

2D and 3D are in the `paper` preset and have not been run.

## Status

| experiment | what it does | `quick` | `paper` |
|---|---|---|---|
| `bias_vs_weight` | bias, spread, coverage vs ρ, four designs | run (9 s) | run (11 min) |
| `misspecification` | six worlds, all estimators | run (30 s) | run (7 min) |
| `uv_divergence` | cut-off dependence, d = 1, 2, 3 | run (6 s) | run (2 min) |
| `pinn_vs_ideal` | trained PINN vs closed form, LM and Adam | run (5 min) | **not run** |
| `pinn_ensemble` | ensemble of PINNs vs predicted bias and spread | run (5 min) | **not run** |
| `adaptive_weights` | which ρ loss balancing ends up at | run (4 min) | **not run** |
| `forward_benchmark` | forward solve, three recipes, 1D–3D | run, 1D only | **not run** |

Everything above was produced on a 2-core CPU sandbox with 4 GB of RAM and no
GPU; times are for that machine. The neural `paper` presets are sized by
estimate for a 6-core laptop with an RTX 3050 (4 GB) and have never been
executed. Treat their settings as a starting point.

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # pick the wheel for your driver
pip install -r requirements.txt
pip install -e .

python -m pytest -q                              # checks of theory and derivatives
python -m idealpinn list
python -m idealpinn all --preset quick --force   # smoke run of everything
python -m idealpinn all --preset paper --only theory --force
python -m idealpinn run pinn_vs_ideal --preset paper
```

`all` skips any experiment that already has results, and the repository ships
reference results, so `--force` is needed to rerun them on your machine.

Outputs go to `results/<experiment>/<preset>/` (`results.json`,
`run_info.json` with the full configuration and environment, figures).
Any preset entry can be overridden: `--set n_datasets=16 group_size=4`.

Step-by-step instructions for an AI agent, including what to do on
out-of-memory and what to report back: [`AGENTS.md`](AGENTS.md).

## Limitations and open questions

* **Linear PDE, one scalar parameter, simple geometry.** The closed form
  relies on linearity. For non-linear PDEs the same construction is only a
  Laplace approximation.
* **The network check is small and 1D**: one data set at full quality, eight
  networks in the ensemble. The central claim "trained PINNs follow the
  tree-level estimator" rests on the `paper` runs that are still to be done.
* **Collocation effects are not quantified.** A network can undercut the
  continuum loss by exploiting the finite collocation set, and its α then
  moves (one case above, +1.3 CRB). At ρ = 1 the PDE residual on fresh points
  was 9 times (with the α warm-up) to about 400 times (an earlier run without
  it, not kept in `results/`) the residual on the training points. The ideal field has integrable
  singularities in its residual at the sensor locations, so part of this may
  be physics and not plain over-fitting; that has not been checked. A sweep
  over `n_colloc` is the next experiment to add.
* **Scan range.** Estimates are found on a grid of ±14 CRB around the truth.
  Rows marked "scan edge" (or `edge > 0.02`) are bounds, not values.
* **Novelty.** That soft-constrained PINN parameter estimates are biased is
  known, and so is the field-theory/kernel reading of the PINN loss (see
  references in `docs/THEORY.md`). What may be new is the closed-form
  estimator with its frequentist properties against the Cramér–Rao bound, the
  identification of ρ, the certification of training by the ideal loss value,
  and the resolution dependence for d ≥ 2. The literature search behind this
  was done from abstracts and summaries, not full texts. A proper search
  is needed before claiming any of it in a paper, in particular against the
  work of Alberts & Bilionis and of Doumèche, Bach, Biau & Boyer.
* **No classical baseline beyond the MLE** (which is the classical baseline
  for this problem, since the forward solution is known in closed form). BiLO
  and Bayesian PINNs are not implemented; `profiled` is the closed-form
  counterpart of a bilevel outer loss with a soft inner fit, not BiLO itself.

## Repository layout

```
idealpinn/exact/        closed-form theory: spectral solution, kernel, estimators, studies
idealpinn/nn/           PyTorch PINNs: analytic-derivative MLP, Levenberg–Marquardt, Adam
idealpinn/experiments/  seven experiments, each with quick and paper presets
tests/                  17 tests, incl. brute-force and Monte-Carlo checks of the formulas
docs/THEORY.md          definitions, derivations, references
results/                reference outputs from the development machine
legacy/                 the original 2025 TensorFlow project (forward PINNs, 1D–3D)
AGENTS.md               run book
```

## Licence

MIT, see `LICENSE`.
