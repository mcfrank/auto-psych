#!/bin/bash
#SBATCH --job-name=paper_parameter_summaries
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# ============================================================================
#  Posterior summaries of the winning and seed models' parameters
#  (scripts/paper/parameter_summaries.py), read from the figure-data jobs'
#  fits on $SCRATCH (no MCMC); the ~100 KB tables are written into the
#  checkout's data/paper/parameter_summaries/.
#
#    cd $HOME/auto-psych && sbatch --output=$SCRATCH/auto-psych/paper/%x-%j.out \
#        scripts/paper/parameter_summaries.sbatch
# ============================================================================
set -euo pipefail
export REPO="${REPO:-$HOME/auto-psych}"
export WORK_ROOT="${WORK_ROOT:-$SCRATCH/auto-psych/paper}"
source "$REPO/scripts/_venv.sh"
export UV_PROJECT_ENVIRONMENT="$(shared_venv_path)"
source "$REPO/scripts/subjective_randomness/slurm/_env.sh"
export PYTENSOR_FLAGS="base_compiledir=${L_SCRATCH:-/tmp}/pytensor/job_${SLURM_JOB_ID}"

cd "$REPO"
echo "[paper] code $(git rev-parse HEAD) $( [[ -z "$(git status --porcelain)" ]] && echo clean || echo DIRTY)"
"$VENV_PY" -m scripts.paper.parameter_summaries \
  --sr-work-dir "$WORK_ROOT/sr_figure_data" \
  --number-game-work-dir "$WORK_ROOT/number_game_figure_data"
