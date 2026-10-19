#!/bin/bash
#SBATCH --partition=normal
#SBATCH --output=%x-%A_%a.out
# ============================================================================
#  One shard of a figure-data job's removed models (submit_figure_data.sh,
#  which sets the array, its resources and MODULE / N_SHARDS). Uses the venv
#  the setup job synced; refuses to run on another commit than it recorded.
# ============================================================================
set -euo pipefail
: "${FIT_WORK_DIR:?set by submit_figure_data.sh}" "${MODULE:?}" "${N_SHARDS:?}"
source "$REPO/scripts/subjective_randomness/slurm/_env.sh"
export PYTENSOR_FLAGS="base_compiledir=${L_SCRATCH:-/tmp}/pytensor/job_${SLURM_JOB_ID}"

cd "$REPO"
recorded="$(cat "$FIT_WORK_DIR/code_commit")"
[[ "$(git rev-parse HEAD)" == "$recorded" ]] || {
  echo "FATAL: the checkout is at $(git rev-parse HEAD), the setup job recorded $recorded" >&2; exit 1; }
echo "[paper] shard $SLURM_ARRAY_TASK_ID of $N_SHARDS, code $recorded"
"$VENV_PY" -m "$MODULE" --work-dir "$FIT_WORK_DIR" \
  --removed-shard "$SLURM_ARRAY_TASK_ID" --removed-shards "$N_SHARDS"
