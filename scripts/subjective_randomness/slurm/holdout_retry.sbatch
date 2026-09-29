#!/bin/bash
#SBATCH --job-name=holdout_retry
#SBATCH --partition=normal
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#
# Runs after an array finishes (afterany). Resumes its failed tasks: a timeout,
# a crash or a node failure gets another attempt from the cell's last finished
# stage; a task that ran out of memory is resubmitted with 128 GB (16 CPUs x
# 8 GB, the normal partition's ceiling). A task the user cancelled is left
# alone. Each resubmission chains its own retry job, up to MAX_RETRY_ROUNDS.
# Cells used to die this way and then silently vanish from every report.
set -euo pipefail
SLURM_DIR="${HOLDOUT_SLURM_DIR:?HOLDOUT_SLURM_DIR is not set}"
source "$SLURM_DIR/_env.sh"
: "${RETRY_ARRAY_ID:?RETRY_ARRAY_ID is not set}"
round=$(( ${RETRY_ROUND:-0} + 1 ))

plan=$(sacct -j "$RETRY_ARRAY_ID" -X -n -P -o JobID,State | "$VENV_PY" "$SLURM_DIR/cell_status.py" retry-plan)
same=$(printf '%s' "$plan" | "$VENV_PY" -c 'import json,sys; print(",".join(map(str, json.load(sys.stdin)["same_memory"])))')
more=$(printf '%s' "$plan" | "$VENV_PY" -c 'import json,sys; print(",".join(map(str, json.load(sys.stdin)["more_memory"])))')
echo "[retry] array $RETRY_ARRAY_ID, round $round: resume [$same], resume with 128 GB [$more]"

if [[ -n "$same" ]]; then
  RETRY_ROUND=$round ARRAY_TASKS="$same" bash "$SLURM_DIR/submit_holdout_test_retest.sh"
fi
if [[ -n "$more" ]]; then
  RETRY_ROUND=$round ARRAY_TASKS="$more" SBATCH_MEM_PER_NODE=128G \
    bash "$SLURM_DIR/submit_holdout_test_retest.sh"
fi
[[ -n "$same$more" ]] || echo "[retry] nothing to resume"
