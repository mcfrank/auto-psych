#!/bin/bash
#SBATCH --job-name=paper_fig3_heldout
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
# ============================================================================
#  Figure 3's rows (a) and (b) on Figure 2's split (fig3_heldout_conditions.py):
#  subjective randomness' leave-one-out and the number game's unseen-item
#  predictions per condition. Reads the pooled fits from the figure data's fit
#  cache; never samples.
#    DOMAIN=subjective-randomness FIT_WORK_DIR=<SR figure data work dir> sbatch scripts/paper/fig3_heldout_conditions.sbatch
#    DOMAIN=number-game FIT_WORK_DIR=<number-game figure data work dir> sbatch scripts/paper/fig3_heldout_conditions.sbatch
# ============================================================================
set -euo pipefail
: "${DOMAIN:?subjective-randomness or number-game}"
: "${FIT_WORK_DIR:?set by the submitting shell}"
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
source "$REPO/scripts/_venv.sh"
export UV_PROJECT_ENVIRONMENT="$(shared_venv_path)"
source "$REPO/scripts/subjective_randomness/slurm/_env.sh"
export PYTENSOR_FLAGS="base_compiledir=${L_SCRATCH:-/tmp}/pytensor/job_${SLURM_JOB_ID}"

cd "$REPO"
echo "[paper] Figure 3 held-out conditions ($DOMAIN), code $(git rev-parse HEAD 2>/dev/null || echo 'not a checkout')"
"$VENV_PY" -m scripts.paper.fig3_heldout_conditions "$DOMAIN" --work-dir "$FIT_WORK_DIR"
