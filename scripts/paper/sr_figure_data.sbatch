#!/bin/bash
#SBATCH --job-name=paper_sr_figure_data
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
# ============================================================================
#  The subjective-randomness data behind the paper's Figures 2 and 3
#  (scripts/paper/sr_figure_data.py): 15 MCMC fits, the three live runs'
#  winners and the four seeds, on each run's final pooled data, at the live
#  run's sampler settings. About 10 GB of fits go to $SCRATCH; the ~1 MB
#  figure data are written into the checkout.
#
#    cd $HOME/auto-psych && sbatch scripts/paper/sr_figure_data.sbatch
#
#  A rerun reuses the fits already in $FIT_WORK_DIR. Then, anywhere:
#    uv run python -m scripts.paper.figures
# ============================================================================
set -euo pipefail
export REPO="${REPO:-$HOME/auto-psych}"
export WORK_ROOT="${WORK_ROOT:-$SCRATCH/auto-psych/paper}"
source "$REPO/scripts/subjective_randomness/slurm/_env.sh"
FIT_WORK_DIR="${FIT_WORK_DIR:-$WORK_ROOT/sr_figure_data}"
export PYTENSOR_FLAGS="base_compiledir=${L_SCRATCH:-/tmp}/pytensor/job_${SLURM_JOB_ID}"

cd "$REPO"
echo "[paper] code $(git rev-parse HEAD) $( [[ -z "$(git status --porcelain)" ]] && echo clean || echo DIRTY)"
uv sync --locked
"$VENV_PY" -m scripts.paper.sr_figure_data --work-dir "$FIT_WORK_DIR"
