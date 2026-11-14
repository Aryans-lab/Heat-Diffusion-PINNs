# Instructions for an AI coding agent (or a human in a hurry)

Goal: reproduce every result of this repository on the owner's PC
(Ryzen 5 5600H, 16 GB RAM, RTX 3050 4 GB) and leave the outputs in `results/`.
Work through the steps in order. Do not skip the checks. Do not edit source
files to make a check pass; report the failure instead.

## 0. Rules

* Run exactly one experiment at a time (GPU memory is 4 GB).
* Never change numbers in `PRESETS` and then report the run as the "paper"
  preset. If something must be reduced, use `--set key=value` and say so; the
  override is recorded in `run_info.json`.
* All commands are run from the repository root with the virtual environment
  active.
* If a step fails, stop, copy the last 40 lines of output, and report.

## 1. Environment

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        Linux/WSL:  source .venv/bin/activate
python -m pip install --upgrade pip
# PyTorch with CUDA (choose the index URL that pytorch.org gives for the installed driver):
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Expected: the last line prints `True` and the GPU name. If it prints `False`
everything still runs, on the CPU, several times slower. Say so in the report.

## 2. Tests (under a minute)

```bash
python -m pytest -q
```

Expected: all tests pass. These include independent checks of the closed-form
theory (brute-force minimisation, direct stochastic simulation) and of the
network derivatives.

## 3. Smoke run (about 15–25 minutes on the CPU, less with a GPU)

```bash
python -m idealpinn all --preset quick --force
```

`--force` is required: the repository ships reference results from the
development machine, and without it every experiment that already has a
`run_info.json` is skipped.

Expected: no line says `already done, skipping`, the final line is
`=== summary: all experiments finished`, and each
`results/<experiment>/quick/` holds `results.json`, `run_info.json` and at
least one `.png`.

## 4. Full runs

Theory (CPU only, about 20 minutes in total):

```bash
python -m idealpinn all --preset paper --only theory --force
```

Neural, one at a time, in this order (the first is the most important):

```bash
python -m idealpinn run pinn_vs_ideal     --preset paper
python -m idealpinn run pinn_ensemble     --preset paper
python -m idealpinn run adaptive_weights  --preset paper
python -m idealpinn run forward_benchmark --preset paper
```

The neural paper presets have **not** been timed on this hardware. Expect
hours, not minutes; `pinn_ensemble` is the longest. Every run prints progress
with elapsed seconds, so extrapolate after the first few lines and report the
estimate before waiting.

`python -m idealpinn all` skips experiments that already have a
`run_info.json` (including the shipped reference results); `--force` reruns
them. `python -m idealpinn run <experiment>` never skips.

### If the GPU runs out of memory

Lower the number of networks trained together, nothing else:

```bash
python -m idealpinn run pinn_ensemble --preset paper --set group_size=4
```

`group_size` changes speed and memory only, not the result.

### If a run is too slow

Allowed reductions, in order of preference, each to be reported:
`n_datasets` (ensemble size), then `n_seeds`, then `dims=[1,2]` for
`forward_benchmark`. Do not reduce `lm_iters`, `adam_steps` or `n_colloc`
without saying that convergence is then not guaranteed.

## 5. What to report back

For each experiment: wall time (`seconds` in `run_info.json`), device, any
`--set` overrides, and the lines of console output that start with four
spaces (they are the result summaries). For `pinn_vs_ideal` and
`pinn_ensemble` quote the `(PINN - tree)/CRB` numbers verbatim. For `pinn_ensemble` also
report how many networks were certified at each ρ, and both the all-network
and the certified-only lines.

Flag, do not hide:

* any estimator with `edge > 0.02` (the estimate left the scan range; the
  number is a bound, not a value);
* `(PINN - tree)/CRB` larger than about 0.5 in magnitude (network not
  converged to the predicted estimator);
* Adam runs whose α is still moving at the last step.

## 6. Layout

```
idealpinn/exact/        closed-form theory (NumPy)
idealpinn/nn/           PINNs (PyTorch): model, Levenberg–Marquardt, Adam
idealpinn/experiments/  one module per experiment: PRESETS, run(), plot()
tests/                  pytest suite
docs/THEORY.md          definitions and derivations
results/                outputs (committed reference runs from the development machine)
legacy/                 the original 2025 TensorFlow project, kept for reference
```
