#!/bin/bash
#SBATCH --job-name=holdout_retry
#SBATCH --partition=mcfrank
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#
# Runs after an array finishes (afterany). Resumes its failed tasks: a timeout,
# a crash or a node failure gets another attempt from the cell's last finished
# stage; a task that ran out of memory is resubmitted with 128 GB (16 CPUs x
# 8 GB, the normal partition's ceiling). A task the user cancelled is left
# alone, and so is a task whose cell already has its holdout.json (Slurm can
# mark a task that finished as OUT_OF_MEMORY or TIMEOUT). Each resubmission
# chains its own retry job, up to MAX_RETRY_ROUNDS.
# Cells used to die this way and then silently vanish from every report.
#
# Retries run on the code the sweep's setup job staged, never on the live
# checkout: this script, _env.sh and the submit script are the staged copies
# in harness_repo, and the resubmission skips the setup job. Shared by the
# literature and the impossible-model sweeps: RETRY_SUBMIT_SCRIPT, exported by
# the sweep's submit script, names the one to resubmit through.
set -euo pipefail
: "${WORK_ROOT:?WORK_ROOT is not set}"
HARNESS_REPO="${HARNESS_REPO:-$WORK_ROOT/harness_repo}"
SLURM_DIR="$HARNESS_REPO/scripts/subjective_randomness/slurm"
[[ -f "$SLURM_DIR/_env.sh" ]] || { echo "ERROR: no staged scripts at $SLURM_DIR (the setup job did not stage this sweep)" >&2; exit 1; }
source "$SLURM_DIR/_env.sh"
: "${RETRY_ARRAY_ID:?RETRY_ARRAY_ID is not set}"
: "${RETRY_SUBMIT_SCRIPT:?RETRY_SUBMIT_SCRIPT is not set: the submit script of the sweep exports it}"
[[ "$RETRY_SUBMIT_SCRIPT" == */* ]] && { echo "ERROR: RETRY_SUBMIT_SCRIPT must be a script name in $SLURM_DIR, got $RETRY_SUBMIT_SCRIPT" >&2; exit 1; }
[[ -f "$SLURM_DIR/$RETRY_SUBMIT_SCRIPT" ]] || { echo "ERROR: no staged $SLURM_DIR/$RETRY_SUBMIT_SCRIPT" >&2; exit 1; }
round=$(( ${RETRY_ROUND:-0} + 1 ))

plan=$(sacct -j "$RETRY_ARRAY_ID" -X -n -P -o JobID,State \
  | "$VENV_PY" "$SLURM_DIR/cell_status.py" retry-plan --work-root "$WORK_ROOT" --gt-models "$GT_MODELS")
same=$(printf '%s' "$plan" | "$VENV_PY" -c 'import json,sys; print(",".join(map(str, json.load(sys.stdin)["same_memory"])))')
more=$(printf '%s' "$plan" | "$VENV_PY" -c 'import json,sys; print(",".join(map(str, json.load(sys.stdin)["more_memory"])))')
echo "[retry] array $RETRY_ARRAY_ID, round $round: resume [$same], resume with 128 GB [$more]"

# ARRAY_MEM is set explicitly for both groups: the environment this job
# inherited may carry an earlier round's value.
if [[ -n "$same" ]]; then
  RETRY_ROUND=$round ARRAY_TASKS="$same" ARRAY_MEM="" \
    bash "$SLURM_DIR/$RETRY_SUBMIT_SCRIPT"
fi
if [[ -n "$more" ]]; then
  RETRY_ROUND=$round ARRAY_TASKS="$more" ARRAY_MEM=128G \
    bash "$SLURM_DIR/$RETRY_SUBMIT_SCRIPT"
fi
[[ -n "$same$more" ]] || echo "[retry] nothing to resume"
