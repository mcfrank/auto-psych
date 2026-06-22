#!/bin/bash
#SBATCH --job-name=impossible_holdout_setup
#SBATCH --partition=normal
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#
# Stage 1/3 of impossible-theory holdout test-retest: build the venv ONCE (so
# array tasks don't race on it) and stage pristine, OFF-the-agent copies of the
# impossible ground-truth recipe.
#
# Unlike the standard holdout (where the GT is a held-out SEED model), the
# impossible GT lives in its own directory OUTSIDE the seed pool. The agent is
# seeded with the full, normal seed set and is *expected to fail* to recover the
# weird generator. The leak risk is therefore not the seed pool but the recipe
# itself: both the impossible model files (e.g. more_heads_more_random.py) and
# their giveaway NAMES in the config would tell the agent the answer outright. So
# we stage two pristine snapshots on scratch, outside any agent's cwd:
#   * $IMPOSSIBLE_MODELS_SRC      — the impossible PyMC generators
#   * $IMPOSSIBLE_CONFIG_SRC      — the config (the array rewrites gt_models_dir
#                                   to the snapshot above and feeds THIS to the
#                                   parent, never the agent's repo copy).
#
# Venv: we do NOT `uv sync` the full project lock — Sherlock compute nodes are
# el7 (glibc 2.17) and the lock pins manylinux_2_28 wheels. We install only what
# the holdout pipeline imports, with --only-binary so uv picks the newest el7
# wheels. Identical to holdout_setup.sbatch.

set -euo pipefail
SLURM_DIR="${HOLDOUT_SLURM_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "$SLURM_DIR/_env.sh"

echo "[setup] building venv at $UV_PROJECT_ENVIRONMENT (python $UV_PYTHON)"
# Reuse only a venv that actually works; otherwise (missing or corrupted)
# rebuild it from scratch.
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
# after tasks have launched.
"$VENV_PY" -m py_compile \
  scripts/subjective_randomness/impossible_holdout_recovery.py \
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

# --- stage the impossible recipe outside any agent's sandbox ---------------
IMPOSSIBLE_MODELS_REL="${IMPOSSIBLE_MODELS_REL:-src/subjective_randomness/impossible_models}"
IMPOSSIBLE_MODELS_SRC="${IMPOSSIBLE_MODELS_SRC:-$WORK_ROOT/impossible_models_src}"
CFG_REL="${CONFIG:-scripts/subjective_randomness/configs/impossible_holdout_recovery.yaml}"
IMPOSSIBLE_CONFIG_SRC="${IMPOSSIBLE_CONFIG_SRC:-$WORK_ROOT/impossible_config_src.yaml}"

echo "[setup] staging impossible models  -> $IMPOSSIBLE_MODELS_SRC"
mkdir -p "$IMPOSSIBLE_MODELS_SRC"
rsync -a "$REPO/$IMPOSSIBLE_MODELS_REL"/ "$IMPOSSIBLE_MODELS_SRC"/

echo "[setup] staging impossible config  -> $IMPOSSIBLE_CONFIG_SRC"
cp "$REPO/$CFG_REL" "$IMPOSSIBLE_CONFIG_SRC"

# Validate that the array's GT list matches the config's gt_models, that the
# seed path matches, and that the snapshot holds every impossible GT file.
SRC_CONFIG="$REPO/$CFG_REL" \
IMPOSSIBLE_MODELS_SRC="$IMPOSSIBLE_MODELS_SRC" \
SEED_MODELS_REL="${SEED_MODELS_REL:-src/pipelines/outer_loop/projects/subjective_randomness/seed_models}" \
IMPOSSIBLE_MODELS_REL="$IMPOSSIBLE_MODELS_REL" \
"$VENV_PY" - <<'PY'
import os, sys, pathlib, yaml

cfg = yaml.safe_load(pathlib.Path(os.environ["SRC_CONFIG"]).read_text())

cfg_gts = sorted((cfg.get("gt_models") or {}).keys())
env_gts = sorted(os.environ.get("GT_MODELS", "").split())
if cfg_gts != env_gts:
    sys.exit(f"GT_MODELS env {env_gts} != config gt_models {cfg_gts} — "
             f"update GT_MODELS in the submit script or the config.")

seed_rel = os.environ.get("SEED_MODELS_REL", "")
if seed_rel and seed_rel != cfg["seed_models_dir"]:
    sys.exit(f"SEED_MODELS_REL env {seed_rel!r} != config seed_models_dir "
             f"{cfg['seed_models_dir']!r}.")

# The impossible recipe must live OUTSIDE the seed pool (that's what makes the
# agent unable to find it); guard against someone pointing gt_models_dir at the
# seed dir by accident.
gt_rel = cfg.get("gt_models_dir")
if gt_rel == cfg["seed_models_dir"]:
    sys.exit(f"config gt_models_dir {gt_rel!r} must differ from seed_models_dir "
             f"— the impossible recipe has to sit outside the agent's seed pool.")
exp_rel = os.environ.get("IMPOSSIBLE_MODELS_REL", "")
if exp_rel and gt_rel != exp_rel:
    sys.exit(f"config gt_models_dir {gt_rel!r} != IMPOSSIBLE_MODELS_REL "
             f"{exp_rel!r} — keep the staged snapshot and the config in sync.")

src = pathlib.Path(os.environ["IMPOSSIBLE_MODELS_SRC"])
missing = [g for g in env_gts if not (src / f"{g}.py").is_file()]
if missing:
    sys.exit(f"impossible snapshot {src} is missing model files: {missing}")
print(f"[setup] validated GT_MODELS={env_gts}; impossible source={src}")
PY

echo "[setup] done."
