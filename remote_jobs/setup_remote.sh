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
# Sherlock provides python/3.12.1 (`module avail python`); other clusters use
# different names (Della: python/3.11.4; Della-GPU: anaconda3 + conda env;
# TACC: python3/3.11; etc). Edit just this line for a new cluster.
PYTHON_MODULE="${PYTHON_MODULE:-python/3.12.1}"
if command -v module >/dev/null 2>&1 || command -v ml >/dev/null 2>&1; then
  ml "$PYTHON_MODULE" 2>/dev/null || module load "$PYTHON_MODULE" 2>/dev/null || {
    echo "setup_remote.sh: failed to load module '$PYTHON_MODULE'." >&2
    echo "  Run 'ml avail python' on this cluster, then set PYTHON_MODULE=<name> and re-run." >&2
    exit 1
  }
fi
# Sanity check: we need >=3.10 for the langchain stack.
PY_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo '0.0')"
PY_MAJOR="${PY_VER%%.*}"
PY_MINOR="${PY_VER##*.}"
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
  echo "setup_remote.sh: python3 reports version $PY_VER; need >=3.10. Fix PYTHON_MODULE and re-run." >&2
  exit 1
fi
echo "setup_remote.sh: using python $PY_VER from $(command -v python3)"

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
