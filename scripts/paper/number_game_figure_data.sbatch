#!/bin/bash
#SBATCH --job-name=paper_number_game_figure_data
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
# ============================================================================
#  The number-game data behind the paper's Figures 2 and 3
#  (scripts/paper/number_game_figure_data.py): each of the three dataset
#  runs' live models and the three seeds on its final data (7,560 responses;
#  35-50 fits a run, about 9 minutes each on a 15-core laptop), and the three
#  winners and the seeds on all runs pooled (22,680; 30 fits), each fitted on
#  all the data and on each of the 4 grouped-CV folds at the runs' sampler
#  settings, plus the records of the runs' 235 removed models and 15 other
#  live models, which the shards of submit_figure_data.sh score first:
#
#    cd $HOME/auto-psych && bash scripts/paper/submit_figure_data.sh number_game
#
#  (submitting this alone raises unless every removed model has a record).
#  The fits go to $SCRATCH; the figure data are written into the checkout. A
#  rerun reuses the fits already in $FIT_WORK_DIR. Then, anywhere:
#    uv run python -m scripts.paper.figures
# ============================================================================
set -euo pipefail
export REPO="${REPO:-$HOME/auto-psych}"
export WORK_ROOT="${WORK_ROOT:-$SCRATCH/auto-psych/paper}"
source "$REPO/scripts/_venv.sh"
export UV_PROJECT_ENVIRONMENT="$(shared_venv_path)"
source "$REPO/scripts/subjective_randomness/slurm/_env.sh"
FIT_WORK_DIR="${FIT_WORK_DIR:-$WORK_ROOT/number_game_figure_data}"
export PYTENSOR_FLAGS="base_compiledir=${L_SCRATCH:-/tmp}/pytensor/job_${SLURM_JOB_ID}"

cd "$REPO"
if [[ -f "$FIT_WORK_DIR/code_commit" && "$(git rev-parse HEAD)" != "$(cat "$FIT_WORK_DIR/code_commit")" ]]; then
  echo "FATAL: the checkout is at $(git rev-parse HEAD), the setup job recorded $(cat "$FIT_WORK_DIR/code_commit")" >&2; exit 1
fi
echo "[paper] code $(git rev-parse HEAD) $( [[ -z "$(git status --porcelain)" ]] && echo clean || echo DIRTY)"
build_shared_venv
"$VENV_PY" -m scripts.paper.number_game_figure_data --work-dir "$FIT_WORK_DIR"
