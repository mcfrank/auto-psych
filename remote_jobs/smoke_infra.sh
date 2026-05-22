#!/usr/bin/env bash
# Drive the infrastructure smoke end-to-end:
#   1. ssh + git pull on the cluster
#   2. submit smoke_infra.slurm
#   3. poll squeue until the job leaves the queue
#   4. rsync the marker file back
#   5. exit 0 iff the marker landed locally
#
# Fast-running (well under a minute on Sherlock if the queue is empty). No Gemini,
# no batch directory, no sweeping -- just confirms ssh + git pull + sbatch +
# venv + rsync are wired correctly.
#
# Usage: ./remote_jobs/smoke_infra.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

if [ -f "$SCRIPT_DIR/.env.local" ]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.env.local"
fi

REMOTE_HOST="${REMOTE_HOST:-sherlock}"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-\$HOME/auto-psych}"
REMOTE_SCRATCH_DIR="${REMOTE_SCRATCH_DIR:-\$SCRATCH/auto-psych}"

POLL_INTERVAL_SEC=10
POLL_MAX_SEC=900   # 15 minutes; smoke jobs should finish far sooner

LOCAL_MARKER_DIR="remote_jobs/logs/smoke"
mkdir -p "$LOCAL_MARKER_DIR"

# 1) git pull on remote so the latest pipeline.slurm + smoke_infra.slurm are present.
echo "==> git pull on ${REMOTE_HOST}:${REMOTE_PROJECT_DIR}"
ssh "$REMOTE_HOST" "cd ${REMOTE_PROJECT_DIR} && git fetch --quiet && git pull --ff-only" \
  || { echo "smoke_infra: remote git pull failed" >&2; exit 1; }

# 2) sbatch
echo "==> submitting smoke_infra.slurm"
SUBMIT_CMD="cd ${REMOTE_PROJECT_DIR} && export REMOTE_PROJECT_DIR=${REMOTE_PROJECT_DIR} REMOTE_SCRATCH_DIR=${REMOTE_SCRATCH_DIR} && sbatch --parsable remote_jobs/smoke_infra.slurm"
JOB_ID="$(ssh "$REMOTE_HOST" "$SUBMIT_CMD" | tr -d '[:space:]')"
JOB_ID="${JOB_ID%%;*}"  # strip ';cluster' suffix if present
if [ -z "$JOB_ID" ] || ! [[ "$JOB_ID" =~ ^[0-9]+$ ]]; then
  echo "smoke_infra: unexpected sbatch reply: '$JOB_ID'" >&2
  exit 1
fi
echo "    job id: $JOB_ID"

# 3) poll squeue until the job leaves
echo "==> polling squeue (every ${POLL_INTERVAL_SEC}s, max ${POLL_MAX_SEC}s)"
elapsed=0
while [ "$elapsed" -lt "$POLL_MAX_SEC" ]; do
  state="$(ssh "$REMOTE_HOST" "squeue -j ${JOB_ID} -h -o '%T' 2>/dev/null || true")"
  state="$(echo "$state" | tr -d '[:space:]')"
  if [ -z "$state" ]; then
    echo "    job $JOB_ID has left the queue (after ${elapsed}s)"
    break
  fi
  echo "    [${elapsed}s] state=${state}"
  sleep "$POLL_INTERVAL_SEC"
  elapsed=$((elapsed + POLL_INTERVAL_SEC))
done
if [ "$elapsed" -ge "$POLL_MAX_SEC" ]; then
  echo "smoke_infra: timeout waiting for job ${JOB_ID}" >&2
  exit 1
fi

# 4) rsync marker + slurm log
echo "==> pulling marker and slurm log"
rsync -avz --include="smoke_infra_${JOB_ID}.ok" --exclude='*' \
  "${REMOTE_HOST}:${REMOTE_SCRATCH_DIR}/smoke/" "${LOCAL_MARKER_DIR}/" || true
rsync -avz --include="auto_psych_smoke_infra-${JOB_ID}.out" \
  --include="auto_psych_smoke_infra-${JOB_ID}.err" --exclude='*' \
  "${REMOTE_HOST}:${REMOTE_SCRATCH_DIR}/logs/" "remote_jobs/logs/" || true

# 5) verify
MARKER_LOCAL="${LOCAL_MARKER_DIR}/smoke_infra_${JOB_ID}.ok"
if [ -f "$MARKER_LOCAL" ]; then
  echo
  echo "==> SMOKE INFRA PASSED"
  echo "    marker: $MARKER_LOCAL"
  if [ -f "remote_jobs/logs/auto_psych_smoke_infra-${JOB_ID}.out" ]; then
    echo "    slurm log:"
    sed 's/^/        /' "remote_jobs/logs/auto_psych_smoke_infra-${JOB_ID}.out"
  fi
  exit 0
else
  echo
  echo "==> SMOKE INFRA FAILED" >&2
  echo "    expected marker: $MARKER_LOCAL" >&2
  if [ -f "remote_jobs/logs/auto_psych_smoke_infra-${JOB_ID}.err" ]; then
    echo "    slurm stderr:" >&2
    sed 's/^/        /' "remote_jobs/logs/auto_psych_smoke_infra-${JOB_ID}.err" >&2
  fi
  exit 1
fi
