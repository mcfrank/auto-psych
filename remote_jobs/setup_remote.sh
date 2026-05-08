#!/usr/bin/env bash
# One-time cluster-side setup: clone (if needed), build a venv, install slim deps,
# create the scratch tree. Run this on the cluster login node, NOT locally.
#
# Usage (on the cluster):
#   git clone <YOUR_REPO_URL> ~/auto-psych  # only if you haven't cloned yet
#   cd ~/auto-psych
#   ./remote_jobs/setup_remote.sh
#
# Re-running is safe: pip install is idempotent and venv creation is skipped if present.

set -euo pipefail

# Resolve repo root from the script's location so this works no matter where it's invoked.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

# ---- Cluster-specific: pick a Python module --------------------------------
# Sherlock has `ml python/3.11`. On other clusters, swap this for the right
# module name (e.g. `module load python/3.11.4` on Della) or skip if Python
# is on $PATH already.
if command -v module >/dev/null 2>&1 || command -v ml >/dev/null 2>&1; then
  ml python/3.11 || module load python/3.11 || true
fi

# Create venv at REPO_DIR/venv (reuse if it already exists).
if [ ! -x "$REPO_DIR/venv/bin/python" ]; then
  python3 -m venv "$REPO_DIR/venv"
fi
# shellcheck disable=SC1091
source "$REPO_DIR/venv/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r requirements-remote.txt

# Pre-create the scratch tree so the slurm template can write logs/results without
# racing on directory creation when many jobs start at once.
SCRATCH_BASE="${SCRATCH:-$HOME/scratch}"
SCRATCH_DIR="${REMOTE_SCRATCH_DIR_OVERRIDE:-$SCRATCH_BASE/auto-psych}"
mkdir -p "$SCRATCH_DIR/logs" "$SCRATCH_DIR/projects"

cat <<EOF

setup_remote.sh: done.
  repo:    $REPO_DIR
  venv:    $REPO_DIR/venv
  scratch: $SCRATCH_DIR

Next steps (one-time):
  - Put GOOGLE_API_KEY into $REPO_DIR/.secrets (one line: GOOGLE_API_KEY=...)
  - Try the infra smoke from your laptop:
      ./remote_jobs/smoke_infra.sh
EOF
