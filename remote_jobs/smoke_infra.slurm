#!/usr/bin/env bash
# Infrastructure smoke: a minimal SLURM job that imports src and writes a marker file.
# No Gemini calls, no batch directory, no sweeping. Exists to verify ssh + git pull +
# sbatch + venv + rsync all work end-to-end.
#
# This file ships its own #SBATCH directives (it is submitted by smoke_infra.sh
# directly, not by submit.py) so it stays small and self-contained.

#SBATCH --job-name=auto_psych_smoke_infra
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --partition=normal

set -euo pipefail

cd "${REMOTE_PROJECT_DIR:-$HOME/auto-psych}"

if command -v module >/dev/null 2>&1 || command -v ml >/dev/null 2>&1; then
  ml python/3.11 || module load python/3.11 || true
fi

# shellcheck disable=SC1091
source "$(pwd)/venv/bin/activate"

SCRATCH_BASE="${SCRATCH:-$HOME/scratch}"
RESULTS_DIR="${REMOTE_SCRATCH_DIR:-$SCRATCH_BASE/auto-psych}"
mkdir -p "$RESULTS_DIR/logs" "$RESULTS_DIR/smoke"

echo "===== smoke_infra job ${SLURM_JOB_ID:-unknown} ====="
echo "Host:    $(hostname)"
echo "Repo:    $(pwd) ($(git rev-parse --short HEAD 2>/dev/null || echo 'nogit'))"
echo "Scratch: $RESULTS_DIR"
echo "Start:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Sanity check: import src so we know the venv is wired and src is on the path.
python3 -c "import sys, src; print('python', sys.version.split()[0]); print('src package OK')"

# Marker file. smoke_infra.sh rsyncs this back and exits 0 only if it appears locally.
MARKER="$RESULTS_DIR/smoke/smoke_infra_${SLURM_JOB_ID:-noid}.ok"
{
  echo "smoke_infra OK"
  echo "job_id=${SLURM_JOB_ID:-unknown}"
  echo "host=$(hostname)"
  echo "commit=$(git rev-parse HEAD 2>/dev/null || echo nogit)"
  echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$MARKER"
echo "Wrote: $MARKER"
echo "End:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
