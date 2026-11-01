#!/bin/bash
#SBATCH --job-name=paper_number_game_pooled_fit
#SBATCH --partition=normal
#SBATCH --array=0-29
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=%x-%A_%a.out
# ============================================================================
#  The number-game figure data's pooled fits, one per task: each run's winner
#  and each seed (6) on all runs' data and on each of its 4 grouped-CV
#  training sets (5), at the pooled sampler settings, into the pooled fit
#  cache (number_game_figure_data.py --pooled-fit-task). The assembly
#  (number_game_figure_data.sbatch) then reads them from the cache. One fit of
#  run 3's winner took 9.5-13 h at these settings in the finals comparison.
#  Refuses to run on another commit than the setup job recorded.
# ============================================================================
set -euo pipefail
: "${FIT_WORK_DIR:?set by the submitting shell}"
source "$REPO/scripts/_venv.sh"
export UV_PROJECT_ENVIRONMENT="$(shared_venv_path)"
source "$REPO/scripts/subjective_randomness/slurm/_env.sh"
export PYTENSOR_FLAGS="base_compiledir=${L_SCRATCH:-/tmp}/pytensor/job_${SLURM_JOB_ID}"

cd "$REPO"
recorded="$(cat "$FIT_WORK_DIR/code_commit")"
[[ "$(git rev-parse HEAD)" == "$recorded" ]] || {
  echo "FATAL: the checkout is at $(git rev-parse HEAD), the setup job recorded $recorded" >&2; exit 1; }
echo "[paper] pooled fit task $SLURM_ARRAY_TASK_ID, code $recorded"
"$VENV_PY" -m scripts.paper.number_game_figure_data --work-dir "$FIT_WORK_DIR" \
  --pooled-fit-task "$SLURM_ARRAY_TASK_ID"
