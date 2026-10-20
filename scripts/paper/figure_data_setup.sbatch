#!/bin/bash
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
# ============================================================================
#  Setup of a figure-data job (submit_figure_data.sh): build the venv once
#  (_venv.sh), before the array's tasks share it, and record the code
#  they must run.
# ============================================================================
set -euo pipefail
: "${FIT_WORK_DIR:?set by submit_figure_data.sh}"
source "$REPO/scripts/subjective_randomness/slurm/_env.sh"
cd "$REPO"
source "$REPO/scripts/paper/_venv.sh"
mkdir -p "$FIT_WORK_DIR"
git rev-parse HEAD > "$FIT_WORK_DIR/code_commit"
echo "[paper] code $(cat "$FIT_WORK_DIR/code_commit") $( [[ -z "$(git status --porcelain)" ]] && echo clean || echo DIRTY)"
