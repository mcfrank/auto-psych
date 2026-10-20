#!/bin/bash
#SBATCH --job-name=srgen_fit
#SBATCH --partition=normal
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --array=1-32
#SBATCH --output=%x_%A_%a.out
#
# Stage 2/4: one MCMC fit per task (line $SLURM_ARRAY_TASK_ID of fits.tsv),
# 4 chains on 4 cores, at the first sampler seed that passes the convergence
# gate; the seed is recorded in results/generalization/fit_seeds/. Most fits
# take 5-40 min; the time limit covers a seed search on the slowest model.
#
# FIT_SEEDS (space-separated) overrides the seed order, e.g. to rerun a fit
# known to stick at the default seed:
#   FIT_SEEDS="6 2 1 20261003" sbatch --array=4,14 fit_array.sbatch
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/_env.sh"

read -r -a seeds <<< "${FIT_SEEDS:-}"
uv run --no-sync python "$ANALYSIS/fit_transfer_models.py" fit \
  --task "$SLURM_ARRAY_TASK_ID" --cache-dir "$FIT_CACHE" ${seeds[@]+--seeds "${seeds[@]}"}
