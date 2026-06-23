# Behavioral analysis (subjective randomness, human runs)

Analysis of the human Prolific data in
`data/results/human_experiment/run{1,2,3}/.../experiment{1,2,3}/data/responses.csv`
(3 chains × 3 rounds × 40 participants × 32 trials).

## Notebooks (Quarto, tidyverse/R)

- `00_load_human_data.qmd` — merge all cells into one trial table with a globally
  unique `participant_uid`; diagnostics of what participants saw (n, collection
  time, stimulus lengths). Saves `data/human_trials.{rds,csv}`.
- `01_response_analysis.qmd` — choice biases (side / n_heads / length /
  alternation, with topline 95% CIs) and stimulus reuse (H/T-consolidated pair
  and single-sequence analyses).
- `02_best_fitting_theories.qmd` — the **mega-analytic model comparison**: reads
  the fit outputs below and reports ELPD-LOO, RMSE, and R². Renders to a
  "run the fit first" notice until those CSVs exist.

## Mega-analytic fit (Python + PyMC)

`fit_mega_models.py` fits one population-level model per theory on the **pooled**
human data and writes `data/mega_model_metrics.csv` and
`data/mega_model_predictions.csv`. It reuses the project harness verbatim
(`src/subjective_randomness/features.py`, `src/models/pymc_inference.py`).

Model set (`mega_models.json`): 4 canonical seed theories + the 4 overall run
winners (run 2 ended in a near-tie → two models). Each `.py` is copied verbatim
so any model-specific `compute_features` is preserved.

### Run on Sherlock (recommended)

```bash
# from the repo root on a login node, after committing your changes:
QUICK=1 sbatch analysis/behavioral/slurm/run_mega_comparison.sbatch   # pre-flight
sbatch analysis/behavioral/slurm/run_mega_comparison.sbatch           # full run
```

The job mirrors `scripts/subjective_randomness/slurm/_env.sh` conventions
(gcc for PyTensor's JIT, uv-managed Python 3.12, venv + caches off `$HOME`).
`.nc` fit caches go to `$WORK_ROOT`; the small result CSVs land in
`analysis/behavioral/data/` for the notebook. Sync/commit those back, then
render `02`.

### Run locally

Needs a PyMC env (not set up by default here). Then:

```bash
python analysis/behavioral/fit_mega_models.py            # full
python analysis/behavioral/fit_mega_models.py --quick    # tiny MCMC smoke test
```

### Sanity check

The harness is the same one the outer loop used, so a per-run subset fit should
reproduce the `model_posterior.json` ELPD-LOO values already in the run
directories (e.g. run 1 / experiment 3).
