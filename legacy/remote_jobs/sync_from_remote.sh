#!/usr/bin/env bash
# Pull batch outputs and slurm logs back from the cluster into the local repo.
#
# Reads remote_jobs/.env.local for REMOTE_HOST and REMOTE_SCRATCH_DIR.
# rsync auth follows your ~/.ssh/config (key + 2FA on Sherlock).
#
# Usage:
#   ./remote_jobs/sync_from_remote.sh                        # all projects, all batches
#   ./remote_jobs/sync_from_remote.sh --project NAME         # only one project
#   ./remote_jobs/sync_from_remote.sh --project NAME --pattern 'batch_2026*'
#   ./remote_jobs/sync_from_remote.sh --logs-only            # just slurm .out/.err files
#
# Examples:
#   ./remote_jobs/sync_from_remote.sh --project subjective_randomness
#   ./remote_jobs/sync_from_remote.sh --pattern 'batch_*nogit'

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

# ---- env ------------------------------------------------------------------
if [ -f "$SCRIPT_DIR/.env.local" ]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.env.local"
else
  echo "Warning: $SCRIPT_DIR/.env.local not found; using defaults." >&2
fi

REMOTE_HOST="${REMOTE_HOST:-sherlock}"
REMOTE_SCRATCH_DIR="${REMOTE_SCRATCH_DIR:-\$SCRATCH/auto-psych}"

PROJECT=""
PATTERN="*"
LOGS_ONLY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --project)   PROJECT="$2"; shift 2;;
    --pattern)   PATTERN="$2"; shift 2;;
    --logs-only) LOGS_ONLY=1; shift;;
    -h|--help)
      sed -n '2,18p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

# ---- pull batch outputs --------------------------------------------------
if [ "$LOGS_ONLY" -eq 0 ]; then
  if [ -n "$PROJECT" ]; then
    REMOTE_BATCHES="${REMOTE_SCRATCH_DIR}/projects/${PROJECT}/batches/${PATTERN}"
    LOCAL_DEST="projects/${PROJECT}/batches/"
    mkdir -p "$LOCAL_DEST"
    echo "Pulling batches: ${REMOTE_HOST}:${REMOTE_BATCHES} -> ${LOCAL_DEST}"
    # -L: dereference any symlinks (mirrors standard_model_2/sync_from_remote.sh)
    # We use a single quoted path so $SCRATCH expands on the remote side; rsync calls
    # ssh under the hood, the remote shell expands env vars.
    rsync -avzL --progress \
      "${REMOTE_HOST}:${REMOTE_BATCHES}" \
      "${LOCAL_DEST}" || true
  else
    echo "Pulling all projects from ${REMOTE_HOST}:${REMOTE_SCRATCH_DIR}/projects/"
    mkdir -p projects/
    rsync -avzL --progress \
      "${REMOTE_HOST}:${REMOTE_SCRATCH_DIR}/projects/" \
      projects/ || true
  fi
fi

# ---- pull slurm logs ------------------------------------------------------
LOCAL_LOGS_DIR="remote_jobs/logs"
mkdir -p "$LOCAL_LOGS_DIR"
echo
echo "Pulling slurm logs: ${REMOTE_HOST}:${REMOTE_SCRATCH_DIR}/logs/ -> ${LOCAL_LOGS_DIR}/"
rsync -avzL --progress \
  --include='*.out' --include='*.err' --include='*/' --exclude='*' \
  "${REMOTE_HOST}:${REMOTE_SCRATCH_DIR}/logs/" \
  "${LOCAL_LOGS_DIR}/" || true

echo
echo "Done. Local state:"
if [ -n "$PROJECT" ] && [ "$LOGS_ONLY" -eq 0 ]; then
  ls -la "projects/${PROJECT}/batches/" 2>/dev/null | tail -10 || true
fi
ls -la "${LOCAL_LOGS_DIR}/" 2>/dev/null | tail -10 || true
