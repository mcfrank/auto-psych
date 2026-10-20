# sr-oct26 generalization analysis on Sherlock

How much of each discovered model's gain over the seeds is shared structure
(pooled winner − best seed) and how much is person-level parameters (full −
pooled), scored three ways, each with a two-way (people × stimulus pairs)
bootstrap:

| task | analysis | scores |
|---|---|---|
| 1, 2 | `within` run 1, run 2 | leave-one-participant-out ELPD on the run's own data |
| 3, 4 | `refit` 1→2, 2→1 | the same on the other run's data, every model refit there |
| 5, 6 | `fixed` 1→2, 2→1 | the source run's fits predict the other run's people as new participants |

## Run

1. The cluster checkout (`~/auto-psych`, `_env.sh`'s `REPO`) must contain
   commit `e007585` (both runs' data): `git pull` there. The analysis folder is
   not committed, so copy it in, leaving out the local caches (the venv and
   uv's Python 3.11 go in `$GROUP_HOME`, the uv and fit caches in `$SCRATCH`):

   ```bash
   rsync -a --exclude mcmc_cache --exclude fits --exclude logs --exclude results --exclude __pycache__ \
     data/results/human_experiment_sr-oct26/analysis/ \
     shr:auto-psych/data/results/human_experiment_sr-oct26/analysis/
   ```

2. On a login node:

   ```bash
   cd ~/auto-psych/data/results/human_experiment_sr-oct26/analysis/slurm
   ./submit.sh            # REPO / WORK_ROOT override the defaults in _env.sh
   ```

   `setup` (venv, model dirs, `fits.tsv`, `scores.tsv`, bootstrap weights) →
   `fit_array` (32 MCMC fits, 4 cores each, seed recorded per fit) →
   `analysis_array` (24 scoring tasks, 2 cores each, no MCMC: per analysis
   one per person-level winner and one for the population-level models) →
   `merge` (each analysis' tables). Roughly 50–90 core-hours, mostly the fits.

3. Copy the results back and report locally (seconds):

   ```bash
   rsync -av shr:auto-psych/data/results/human_experiment_sr-oct26/analysis/results/generalization/ \
     data/results/human_experiment_sr-oct26/analysis/results/generalization/
   uv run python data/results/human_experiment_sr-oct26/analysis/report_generalization.py
   ```

## If something fails

- A fit that converges at none of its 13 seeds fails its array task with the
  seeds tried; the analyses then never start (`afterok`). Rerun that task with
  more seeds: `sbatch --array=<i> fit_array.sbatch` after adding seeds to
  `Fit.seeds` in `fit_transfer_models.py`, then submit `analysis_array.sbatch` and `merge.sbatch` (`--dependency=afterok`).
- An analysis task refuses a fit without a recorded seed, or one whose fit no
  longer passes the convergence gate.
