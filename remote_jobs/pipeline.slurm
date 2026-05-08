#!/usr/bin/env bash
# SLURM template for an auto-psych pipeline run.
#
# This file has NO #SBATCH directives on purpose: all resources (--time, --cpus-per-task,
# --mem, --partition, --output, --error, --job-name) are passed by remote_jobs/submit.py
# from the YAML manifest's `slurm:` block. That keeps one .slurm file generic for every
# matrix cell.
#
# Usage (driven by submit.py, not by hand):
#   sbatch \
#     --job-name=...  --output=...  --error=...  --time=...  --cpus-per-task=...  --mem=...  --partition=... \
#     remote_jobs/pipeline.slurm \
#       --project subjective_randomness \
#       --runs 5 \
#       --mode simulated_participants_nobrowser \
#       --ground-truth-model alternation \
#       --n-participants 5 \
#       --max-retries 3
#
# All args after `pipeline.slurm` are passed straight through to run_pipeline.py.
#
# Cluster-specific bits (flagged here so a new cluster only needs to edit a couple of lines):
#   1) Module load: PYTHON_MODULE defaults to python/3.12.1 (Sherlock's name as of 2026-05).
#      Other clusters use different names (Della: python/3.11.4, TACC: python3/3.11, etc).
#      To override per-cluster without editing this file, export PYTHON_MODULE in
#      remote_jobs/.env.local; the submitter forwards it through the sbatch environment.
#   2) Scratch dir: this script reads $REMOTE_SCRATCH_DIR if exported by submit.py
#      (preferred), else falls back to $SCRATCH/auto-psych. On Sherlock $SCRATCH is
#      defined automatically; on other clusters use $WORK, $GROUP_HOME, etc. and set
#      REMOTE_SCRATCH_DIR in remote_jobs/.env.local.

set -euo pipefail

# ---- Resolve repo root ------------------------------------------------------
# REMOTE_PROJECT_DIR is exported by the submitter (or by .env.local in interactive use).
# Otherwise default to $HOME/auto-psych.
cd "${REMOTE_PROJECT_DIR:-$HOME/auto-psych}"

# ---- Cluster-specific: load Python module -----------------------------------
PYTHON_MODULE="${PYTHON_MODULE:-python/3.12.1}"
if command -v module >/dev/null 2>&1 || command -v ml >/dev/null 2>&1; then
  ml "$PYTHON_MODULE" 2>/dev/null || module load "$PYTHON_MODULE" 2>/dev/null || true
fi

# ---- Scratch tree (results land here, then sync_from_remote.sh pulls back) --
SCRATCH_BASE="${SCRATCH:-$HOME/scratch}"
RESULTS_DIR="${REMOTE_SCRATCH_DIR:-$SCRATCH_BASE/auto-psych}"
mkdir -p "$RESULTS_DIR/logs" "$RESULTS_DIR/projects"

# ---- Activate venv (created by setup_remote.sh) -----------------------------
# shellcheck disable=SC1091
source "$(pwd)/venv/bin/activate"

# ---- Load API keys without printing them ------------------------------------
# .secrets is a simple KEY=VAL file. Export each line as an env var.
if [ -f .secrets ]; then
  set -a
  # shellcheck disable=SC1091
  source .secrets
  set +a
fi

# ---- Tell run_pipeline to write all run artifacts to scratch ----------------
export PIPELINE_PROJECTS_DIR="$RESULTS_DIR/projects"

echo "===== SLURM job ${SLURM_JOB_ID:-unknown} (auto-psych pipeline) ====="
echo "Host:       $(hostname)"
echo "Repo:       $(pwd) ($(git rev-parse --short HEAD 2>/dev/null || echo 'nogit'))"
echo "Scratch:    $RESULTS_DIR"
echo "Projects:   $PIPELINE_PROJECTS_DIR"
echo "Args:       $*"
echo "Start:      $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo

python3 run_pipeline.py "$@"

echo
echo "End:        $(date -u +%Y-%m-%dT%H:%M:%SZ)"
