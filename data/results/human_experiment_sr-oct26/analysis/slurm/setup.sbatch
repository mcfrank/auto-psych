#!/bin/bash
#SBATCH --job-name=srgen_setup
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --output=%x_%j.out
#
# Stage 1/4: sync the venv, write the model-set dirs, the fit manifest
# (fits.tsv, 32 fits) and the scoring tasks (scores.tsv, 24), and draw each
# run's two-way bootstrap weights once.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/_env.sh"

uv sync --locked
uv run python "$ANALYSIS/fit_transfer_models.py" assemble
# The arrays are sized to these manifests: stop if they disagree.
G="$ANALYSIS/results/generalization"
[[ "$(wc -l < "$G/fits.tsv")" -eq 32 ]] || { echo "fits.tsv has $(wc -l < "$G/fits.tsv") lines; fit_array.sbatch expects 32" >&2; exit 1; }
[[ "$(wc -l < "$G/scores.tsv")" -eq 24 ]] || { echo "scores.tsv has $(wc -l < "$G/scores.tsv") lines; analysis_array.sbatch expects 24" >&2; exit 1; }
for run in 1 2; do
  uv run python "$ANALYSIS/generalization_analysis.py" --analysis within --source "$run" --target "$run" --weights-only
done
