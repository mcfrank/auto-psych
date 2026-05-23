#!/usr/bin/env bash
# End-to-end smoke: submit a tiny ground-truth-model run via submit.py, poll until done,
# sync the batch back, and run all six validators on the rehydrated run dir.
#
# Real Gemini API calls happen for theory/design/interpret. The collect step uses the
# deterministic 'alternation' ground-truth model, so no LLM steering / browser / Firebase.
#
# Exits 0 only if every validator passes on the synced run.
#
# Usage:
#   ./remote_jobs/smoke_e2e.sh            # uses remote_jobs/jobs/smoke_e2e.yaml
#   ./remote_jobs/smoke_e2e.sh --allow-dirty
#
# Pre-reqs:
#   - remote_jobs/.env.local set up
#   - $REMOTE_PROJECT_DIR/.secrets contains GOOGLE_API_KEY
#   - HEAD is committed and pushed (unless --allow-dirty)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

if [ -f "$SCRIPT_DIR/.env.local" ]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.env.local"
fi
REMOTE_HOST="${REMOTE_HOST:-sherlock}"

ALLOW_DIRTY_FLAG=""
if [ "${1:-}" = "--allow-dirty" ]; then
  ALLOW_DIRTY_FLAG="--allow-dirty"
fi

MANIFEST="remote_jobs/jobs/smoke_e2e.yaml"
PROJECT="subjective_randomness"

PYTHON="$REPO_DIR/venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON="python3"
fi

POLL_INTERVAL_SEC=30
POLL_MAX_SEC=3600   # 1 hour ceiling; e2e smoke usually finishes in 5-10 min

# 1) Submit and capture job ids
echo "==> submitting $MANIFEST"
SUBMIT_OUT="$(mktemp -t smoke_e2e_submit.XXXXXX)"
if ! "$PYTHON" remote_jobs/submit.py "$MANIFEST" $ALLOW_DIRTY_FLAG 2>&1 | tee "$SUBMIT_OUT"; then
  echo "smoke_e2e: submit.py failed" >&2
  rm -f "$SUBMIT_OUT"
  exit 1
fi
# Parse "submitted <jobid>" lines for job ids (one per cell; smoke_e2e.yaml has 1 cell)
JOB_IDS=$(awk '/^submitted [0-9]+/{print $2}' "$SUBMIT_OUT" | xargs)
rm -f "$SUBMIT_OUT"
if [ -z "$JOB_IDS" ]; then
  echo "smoke_e2e: could not parse a job id from submit.py output" >&2
  exit 1
fi
echo "==> submitted job ids: $JOB_IDS"

# 2) Poll until all job ids leave the queue
echo "==> polling squeue every ${POLL_INTERVAL_SEC}s (max ${POLL_MAX_SEC}s)"
elapsed=0
while [ "$elapsed" -lt "$POLL_MAX_SEC" ]; do
  REMAINING="$(ssh "$REMOTE_HOST" "squeue -h -j $(echo $JOB_IDS | tr ' ' ',') -o '%i %T' 2>/dev/null || true")"
  REMAINING="$(echo "$REMAINING" | tr -d '[:space:]')"
  if [ -z "$REMAINING" ]; then
    echo "    all jobs done after ${elapsed}s"
    break
  fi
  STATES="$(ssh "$REMOTE_HOST" "squeue -h -j $(echo $JOB_IDS | tr ' ' ',') -o '%i=%T'" 2>/dev/null || true)"
  echo "    [${elapsed}s] still queued/running: $STATES"
  sleep "$POLL_INTERVAL_SEC"
  elapsed=$((elapsed + POLL_INTERVAL_SEC))
done
if [ "$elapsed" -ge "$POLL_MAX_SEC" ]; then
  echo "smoke_e2e: timeout waiting for jobs $JOB_IDS" >&2
  exit 1
fi

# 3) Sync results back
echo "==> syncing batches and slurm logs back"
"$SCRIPT_DIR/sync_from_remote.sh" --project "$PROJECT"

# 4) Locate the latest batch we just pulled
LATEST_BATCH="$(ls -1d "projects/${PROJECT}/batches/batch_"* 2>/dev/null | sort | tail -n1 || true)"
if [ -z "$LATEST_BATCH" ] || [ ! -d "$LATEST_BATCH" ]; then
  echo "smoke_e2e: no batch directory found under projects/${PROJECT}/batches/ after sync" >&2
  echo "          (check remote_jobs/logs/auto_psych_smoke_e2e* for the slurm error)" >&2
  exit 1
fi
RUN_DIR="$LATEST_BATCH/run1"
if [ ! -d "$RUN_DIR" ]; then
  echo "smoke_e2e: expected run1/ under $LATEST_BATCH but it's missing" >&2
  exit 1
fi
echo "==> validating $RUN_DIR"

# 5) Run every agent validator over the rehydrated run dir
"$PYTHON" - "$RUN_DIR" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from src.validation.validators import AGENT_VALIDATORS

run_dir = Path(sys.argv[1])
fails = []
for agent_key, fn in AGENT_VALIDATORS.items():
    v = fn(run_dir)
    flag = "PASS" if v.ok else "FAIL"
    print(f"{agent_key}: {flag} - {v.message}")
    if not v.ok:
        fails.append(agent_key)

if fails:
    print(f"\nsmoke_e2e: FAILED -- {len(fails)} validator(s) did not pass: {fails}", file=sys.stderr)
    sys.exit(1)
print("\nsmoke_e2e: PASSED -- all six validators ok.")
PY
