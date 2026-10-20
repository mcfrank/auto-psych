#!/bin/bash
#SBATCH --job-name=srgen_score
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --array=1-24
#SBATCH --output=%x_%A_%a.out
#
# Stage 3/4: one scoring task per line of scores.tsv (written by setup): per
# analysis (leave-one-participant-out within each run, refit and fixed-
# parameter transfer each way), one task per person-level winner and one for
# the seven population-level models together, each with the two-way bootstrap
# (1,000 replicates). Fits are loaded from the cache at their recorded seeds
# (no MCMC). About 5-20 min per task; results go to results/generalization/parts/.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/_env.sh"
# The prediction is single-threaded; the bootstrap's matrix products use BLAS.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

LINE="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$ANALYSIS/results/generalization/scores.tsv")"
[[ -n "$LINE" ]] || { echo "no line $SLURM_ARRAY_TASK_ID in scores.tsv" >&2; exit 1; }
IFS=$'\t' read -r analysis source target part models <<< "$LINE"
IFS=',' read -r -a model_list <<< "$models"
echo "[score] $analysis run $source on run $target, part $part: ${model_list[*]}"
uv run --no-sync python "$ANALYSIS/generalization_analysis.py" \
  --analysis "$analysis" --source "$source" --target "$target" \
  --part "$part" --models "${model_list[@]}" --cache-dir "$FIT_CACHE"
