#!/bin/bash
#SBATCH --job-name=paper_rsa_recovery_trajectory
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=%x-%A_%a.out
# ============================================================================
#  Reference games: every recovery cell's best model's distance to the ground
#  truth at each step (scripts/paper/rsa_recovery_trajectory.py).
#    MODE=setup     sbatch -t 0:30:00 rsa_recovery_trajectory.sbatch
#    MODE=fit       sbatch -t <limit> --array=0-<tasks-1> rsa_recovery_trajectory.sbatch
#    MODE=assemble  sbatch -t 0:30:00 --dependency=afterok:<fit> rsa_recovery_trajectory.sbatch
#  Trial-level data stay in $WORK_ROOT on scratch; only the distances come
#  back, to $REPO/data/paper/rsa_recovery_trajectory.json.
# ============================================================================
set -euo pipefail
export REPO="${REPO:-${SLURM_SUBMIT_DIR:-$HOME/auto-psych}}"
export WORK_ROOT="${WORK_ROOT:-${SCRATCH:?}/auto-psych/rsa_recovery_trajectory}"
source "$REPO/scripts/rsa/slurm/_env.sh"
MODE="${MODE:?set MODE=setup, fit or assemble}"

cd "$REPO"
echo "[paper] rsa recovery trajectory $MODE ${SLURM_ARRAY_TASK_ID:-}, code $(git rev-parse HEAD)"
"$VENV_PY" -m scripts.paper.rsa_recovery_trajectory --mode "$MODE" --work-dir "$WORK_ROOT/work" \
  ${SLURM_ARRAY_TASK_ID:+--task "$SLURM_ARRAY_TASK_ID"}
