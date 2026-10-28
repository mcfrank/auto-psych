#!/bin/bash
#SBATCH --job-name=paper_rsa_final_parameters
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
# ============================================================================
#  Posterior summaries of the three reference-game finals' parameters
#  (scripts/paper/rsa_final_parameters.py): each final refitted on the
#  pre-registration's pooled data with the frozen fits' settings; the tables
#  are written into the checkout's data/paper/parameter_summaries/rsa_finals/.
#
#    cd $HOME/auto-psych && sbatch --output=$SCRATCH/auto-psych/paper/%x-%j.out \
#        scripts/paper/rsa_final_parameters.sbatch
# ============================================================================
set -euo pipefail
export REPO="${REPO:-$HOME/auto-psych}"
export WORK_ROOT="${WORK_ROOT:-$SCRATCH/auto-psych/rsa_final_parameters}"
PREREG_ROOT="${PREREG_ROOT:-$SCRATCH/auto-psych/rsa_prereg}"
source "$REPO/scripts/rsa/slurm/_env.sh"

cd "$REPO"
echo "[paper] code $(git rev-parse HEAD) $( [[ -z "$(git status --porcelain)" ]] && echo clean || echo DIRTY)"
"$VENV_PY" -m scripts.paper.rsa_final_parameters \
  --pooled "$PREREG_ROOT/pool/pooled.csv" --work-dir "$WORK_ROOT"
