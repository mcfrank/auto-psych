#!/bin/bash
#SBATCH --job-name=holdout_setup
#SBATCH --partition=normal
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#
# Stage 1/3: build the venv ONCE (so array tasks don't race on it) and stage a
# pristine snapshot of the seed models on scratch.
#
# We do NOT `uv sync` the full project lock: Sherlock compute nodes are el7
# (glibc 2.17), and the lock pins the latest scipy/pandas/matplotlib whose wheels
# are manylinux_2_28 (glibc >= 2.28) — uv would fall back to slow, fragile source
# builds. Instead we install only what the holdout pipeline imports, with
# --only-binary so uv picks the newest versions that HAVE el7 wheels. pymc/
# pytensor/arviz are pinned to the project's 5.x-era API (pure-Python wheels).

set -euo pipefail
SLURM_DIR="${HOLDOUT_SLURM_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "$SLURM_DIR/_env.sh"

echo "[setup] building venv at $UV_PROJECT_ENVIRONMENT (python $UV_PYTHON)"
# Reuse only a venv that actually works; otherwise (missing or corrupted, e.g.
# a half-written numpy from a clobbered shared env) rebuild it from scratch.
if [[ -x "$VENV_PY" ]] && "$VENV_PY" -c "import numpy" >/dev/null 2>&1; then
  echo "[setup] reusing existing venv"
else
  uv venv --clear --python "$UV_PYTHON" "$UV_PROJECT_ENVIRONMENT"
fi

# Only-binary: never build from source (el7 can't build the scientific stack and
# doesn't need to — older wheel versions work fine). Idempotent on re-run.
uv pip install --python "$UV_PROJECT_ENVIRONMENT" --only-binary :all: \
  "pymc==5.28.5" "pytensor==2.38.3" "arviz<1" \
  numpy scipy pyyaml tyro pyprojroot matplotlib

# Preflight: a coding-agent credential must be available or every run will fail
# the moment it spawns an agent.
if [[ -z "${GOOGLE_API_KEY:-}${GEMINI_API_KEY:-}" ]]; then
  echo "WARNING: no GOOGLE_API_KEY / GEMINI_API_KEY found (env or $REPO/.secrets)." >&2
  echo "         opencode+gemini runs will fail. Set it before the array runs." >&2
fi

# Fail fast: syntax-check the scripts the array/analysis jobs run, and import the
# whole holdout chain so a missing dependency aborts here (afterok) rather than
# after 15 tasks have launched.
"$VENV_PY" -m py_compile \
  scripts/subjective_randomness/holdout_recovery.py \
  scripts/subjective_randomness/holdout_test_retest.py \
  src/subjective_randomness/holdout_recovery.py
"$VENV_PY" - <<'PY'
import sys
from pyprojroot import here
sys.path.insert(0, str(here()))
import src.subjective_randomness.holdout_recovery        # noqa: F401
import src.pipelines.outer_loop.orchestrator             # noqa: F401
import src.pipelines.inner_loop.run                      # noqa: F401
print("[setup] import chain OK")
PY

# Stage a pristine seed/GT-model snapshot outside any agent's sandbox.
SEED_MODELS_REL="${SEED_MODELS_REL:-src/pipelines/outer_loop/projects/subjective_randomness/seed_models}"
GT_MODELS_SRC="${GT_MODELS_SRC:-$WORK_ROOT/gt_models_src}"
echo "[setup] staging pristine seed/GT models -> $GT_MODELS_SRC"
mkdir -p "$GT_MODELS_SRC"
rsync -a "$REPO/$SEED_MODELS_REL"/ "$GT_MODELS_SRC"/

# Validate that the array's GT list / seed path match the config and that the
# snapshot contains every ground-truth model file.
CFG_REL="${CONFIG:-scripts/subjective_randomness/configs/holdout_recovery.yaml}"
SRC_CONFIG="$REPO/$CFG_REL" GT_MODELS_SRC="$GT_MODELS_SRC" SEED_MODELS_REL="$SEED_MODELS_REL" \
"$VENV_PY" - <<'PY'
import os, sys, pathlib, yaml

cfg = yaml.safe_load(pathlib.Path(os.environ["SRC_CONFIG"]).read_text())
cfg_gts = sorted((cfg.get("gt_models") or {}).keys())
env_gts = sorted(os.environ.get("GT_MODELS", "").split())
if cfg_gts != env_gts:
    sys.exit(f"GT_MODELS env {env_gts} != config gt_models {cfg_gts} — "
             f"update GT_MODELS in the submit script or the config.")

rel, seed_rel = cfg["seed_models_dir"], os.environ.get("SEED_MODELS_REL", "")
if seed_rel and seed_rel != rel:
    sys.exit(f"SEED_MODELS_REL env {seed_rel!r} != config seed_models_dir {rel!r}.")

src = pathlib.Path(os.environ["GT_MODELS_SRC"])
missing = [g for g in env_gts if not (src / f"{g}.py").is_file()]
if missing:
    sys.exit(f"GT snapshot {src} is missing model files: {missing}")
print(f"[setup] validated GT_MODELS={env_gts}; GT source={src}")
PY

echo "[setup] done."
