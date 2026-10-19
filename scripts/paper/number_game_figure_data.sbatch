#!/bin/bash
#SBATCH --job-name=paper_number_game_figure_data
#SBATCH --partition=normal
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
# ============================================================================
#  The number-game data behind the paper's Figures 2 and 3
#  (scripts/paper/number_game_figure_data.py): the dataset run's four live
#  models, three seeds and 84 models it removed on the way (pruned, retired
#  or dropped) on its final data (7,560 responses), each fitted on all of
#  them and on each of the 4 grouped-CV folds (455 fits), at the run's
#  sampler settings. The first 35 took 9 minutes on a 15-core laptop. The
#  fits go to $SCRATCH; the figure data are written into the checkout.
#
#    cd $HOME/auto-psych && sbatch scripts/paper/number_game_figure_data.sbatch
#
#  A rerun reuses the fits already in $FIT_WORK_DIR. Then, anywhere:
#    uv run python -m scripts.paper.figures
# ============================================================================
set -euo pipefail
export REPO="${REPO:-$HOME/auto-psych}"
export WORK_ROOT="${WORK_ROOT:-$SCRATCH/auto-psych/paper}"
source "$REPO/scripts/subjective_randomness/slurm/_env.sh"
FIT_WORK_DIR="${FIT_WORK_DIR:-$WORK_ROOT/number_game_figure_data}"
export PYTENSOR_FLAGS="base_compiledir=${L_SCRATCH:-/tmp}/pytensor/job_${SLURM_JOB_ID}"

cd "$REPO"
echo "[paper] code $(git rev-parse HEAD) $( [[ -z "$(git status --porcelain)" ]] && echo clean || echo DIRTY)"
uv sync --locked
"$VENV_PY" -m scripts.paper.number_game_figure_data --work-dir "$FIT_WORK_DIR"
