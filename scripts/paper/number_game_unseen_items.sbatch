#!/bin/bash
#SBATCH --job-name=paper_number_game_unseen_items
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
# ============================================================================
#  The pooled number-game winners and seeds scored on the Bigelow & Piantadosi
#  responses no run drew (number_game_unseen_items.py). Reads the pooled fits
#  from the figure data's fit cache; never samples.
#    FIT_WORK_DIR=<figure data work dir> sbatch scripts/paper/number_game_unseen_items.sbatch
# ============================================================================
set -euo pipefail
: "${FIT_WORK_DIR:?set by the submitting shell}"
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/scripts/_venv.sh"
export UV_PROJECT_ENVIRONMENT="$(shared_venv_path)"
source "$REPO/scripts/subjective_randomness/slurm/_env.sh"
export PYTENSOR_FLAGS="base_compiledir=${L_SCRATCH:-/tmp}/pytensor/job_${SLURM_JOB_ID}"

cd "$REPO"
echo "[paper] unseen items, code $(git rev-parse HEAD)"
"$VENV_PY" -m scripts.paper.number_game_unseen_items --work-dir "$FIT_WORK_DIR"
