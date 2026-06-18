#!/bin/bash
#SBATCH --job-name=holdout_test_retest
#SBATCH --partition=normal
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#
# Stage 3/3: aggregate the repeats into a test-retest reliability summary.
# Submitted with `--dependency=afterok:<array_job_id>` so it runs once the
# array finishes (use afterany if you want it to run even when a repeat fails).

set -euo pipefail
SLURM_DIR="${HOLDOUT_SLURM_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "$SLURM_DIR/_env.sh"

"$VENV_PY" scripts/subjective_randomness/holdout_test_retest.py \
  --runs-root "$WORK_ROOT" \
  --out    "$WORK_ROOT/test_retest.json" \
  --csv    "$WORK_ROOT/test_retest.csv" \
  --figure "$WORK_ROOT/test_retest.png"

echo "[analysis] wrote $WORK_ROOT/test_retest.{json,csv,png}"
