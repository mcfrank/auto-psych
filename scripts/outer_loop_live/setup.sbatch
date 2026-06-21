#!/bin/bash
#SBATCH --job-name=outer_live_setup
#SBATCH --partition=normal
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#
# One-time setup for the LIVE outer-loop harness. Run this ONCE (per WORK_ROOT)
# before submitting runs:
#   1. build the shared uv venv (off $HOME), el7-safe binary wheels only;
#   2. install the Firebase Functions deps + warm the firebase CLI in the cache;
#   3. preflight the live import chain so a missing dep fails HERE, not mid-run.
#
# We do NOT `uv sync` the full project lock: Sherlock compute nodes are el7
# (glibc 2.17) and the lock pins newest scipy/pandas/etc whose wheels are
# manylinux_2_28 (glibc >= 2.28) -> uv would fall back to slow, fragile source
# builds. Instead install only what the LIVE pipeline imports, with
# --only-binary, so uv picks the newest versions that HAVE el7 wheels.

set -euo pipefail
SLURM_DIR="${OUTER_LIVE_SLURM_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "$SLURM_DIR/_env.sh"
cd "$REPO"

echo "[setup] building venv at $UV_PROJECT_ENVIRONMENT (python $UV_PYTHON)"
if [[ -x "$VENV_PY" ]] && "$VENV_PY" -c "import numpy, pymc, tyro, requests, pydantic" >/dev/null 2>&1; then
  echo "[setup] reusing existing venv"
else
  uv venv --clear --python "$UV_PYTHON" "$UV_PROJECT_ENVIRONMENT"
fi

# el7 wheels only (never build from source). The PyMC stack is pinned to the
# project's 5.x API (pure-Python wheels). langchain / google-generativeai /
# torch are intentionally OMITTED — a live human run imports them lazily only on
# the Gemini-steering / open-participant paths it never takes. The live pipeline
# does NO server-side Firestore access either: participant data flows through the
# /submit and /results Cloud Functions (functions/index.js), so google-cloud-*
# is not needed here. (The monitor reads Firestore, but it runs off-cluster.)
uv pip install --python "$UV_PROJECT_ENVIRONMENT" --only-binary :all: \
  "pymc==5.28.5" "pytensor==2.38.3" "arviz<1" \
  numpy scipy pyyaml tyro pyprojroot requests pydantic matplotlib

# Firebase Functions deps. The deploy auto-installs these (per worktree) too,
# but doing it once here warms the npm cache (off $HOME) and validates npm.
npm --prefix "$REPO/functions" install
# Warm the firebase CLI into the npm cache (deploy uses `firebase` if on PATH,
# else `npx -y firebase-tools`).
npx -y firebase-tools --version || true

# Credential preflight — warn (don't fail) so the venv still builds.
[[ -n "${PROLIFIC_API_TOKEN:-}" ]] || \
  echo "WARNING: PROLIFIC_API_TOKEN not set (needed for --prolific-mode test/live)." >&2
[[ -n "${FIREBASE_TOKEN:-}" ]] || \
  echo "WARNING: FIREBASE_TOKEN not set (needed for --deploy-target firebase). Run 'firebase login:ci' and add it to $REPO/.secrets." >&2

# Import the live entry point + collect + inner-loop orchestrator: a missing
# dependency aborts the setup job (afterok) instead of after a run has started.
"$VENV_PY" - <<'PY'
import sys
from pyprojroot import here
sys.path.insert(0, str(here()))
import src.pipelines.outer_loop.run               # noqa: F401  live entry point
import src.pipelines.outer_loop.collect           # noqa: F401
import src.pipelines.outer_loop.deployment        # noqa: F401
import src.pipelines.inner_loop.pymc_orchestrator # noqa: F401
print("[setup] live import chain OK")
PY

echo "[setup] done."
