#!/bin/bash
#SBATCH --job-name=holdout_recovery
#SBATCH --partition=normal
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.out
#
# Stage 2/3: one array task per (repeat, ground-truth model). With R repeats and
# G ground truths the submit script sizes the array to R*G; task T maps to
#   repeat = (T-1)/G + 1   (distinct --seed per repeat -> different synthetic data)
#   gt     = GT_MODELS[(T-1)%G]
#
# LEAK PREVENTION (opencode has no read sandbox and runs with cwd = its repo):
#   * each task runs from its OWN rsync copy of the repo on $SCRATCH;
#   * the copy EXCLUDES ground_truth_models.py / evaluate_recovery.py and has
#     THIS task's held-out seed-model file deleted, so the GT recipe is not on
#     the agent's disk at all;
#   * the parent reads the GT generator via --gt-models-dir from the pristine
#     snapshot staged by the setup job ($WORK_ROOT/gt_models_src), which sits
#     outside every agent's cwd. Seeding still excludes the GT (exclusion keys
#     off the seed manifest, which the copy retains), so removing the file is safe;
#   * opencode read/glob/grep are additionally denied on the off-limits paths.
# The experiment tree (results-root) lives INSIDE the agent's cwd (the repo copy)
# so opencode needs no external-directory grant for it — exactly like the normal
# pipeline. The MCMC cache + holdout.{json,csv,png} live at $WORK_ROOT/run<r>/<gt>/.

set -euo pipefail
SLURM_DIR="${HOLDOUT_SLURM_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "$SLURM_DIR/_env.sh"

# The coding agents inherit this shell's environment. _env.sh exports
# UV_PROJECT_ENVIRONMENT pointing at the SHARED per-run venv; agents invoke
# `uv run python ...` to test their work, and each such call re-syncs that venv
# against the full project lock — wiping our installed packages mid-run and
# breaking every task. The pipeline uses $VENV_PY (an explicit interpreter path,
# already resolved by _env.sh), so drop the UV_* vars here: a stray agent
# `uv run` then targets a throwaway .venv in its own repo copy, never the
# shared environment.
unset UV_PROJECT_ENVIRONMENT UV_CACHE_DIR UV_PYTHON

# --- task -> (repeat, ground-truth model) ----------------------------------
BASE_SEED="${BASE_SEED:-0}"
read -r -a GTS <<< "${GT_MODELS:-bayesian_diagnosticity encoding_compressibility prototype_similarity window_typicality}"
N_GTS="${N_GTS:-${#GTS[@]}}"
SEED_MODELS_REL="${SEED_MODELS_REL:-src/pipelines/outer_loop/projects/subjective_randomness/seed_models}"
CONFIG="${CONFIG:-scripts/subjective_randomness/configs/holdout_recovery.yaml}"
GT_MODELS_SRC="${GT_MODELS_SRC:-$WORK_ROOT/gt_models_src}"  # pristine GT snapshot

TASK="${SLURM_ARRAY_TASK_ID:-1}"
REPEAT=$(( (TASK - 1) / N_GTS + 1 ))
GT="${GTS[$(( (TASK - 1) % N_GTS ))]}"
SEED=$(( BASE_SEED + REPEAT ))

RUN_DIR="$WORK_ROOT/run${REPEAT}/${GT}"
RUN_REPO="$RUN_DIR/repo"
mkdir -p "$RUN_DIR"

# Per-task opencode state: opencode keeps its session SQLite DB under
# XDG_DATA_HOME (default ~/.local/share/opencode). Concurrent array tasks
# sharing one DB corrupt it ("SQLITE_CORRUPT: database disk image is
# malformed"). Isolate every task's XDG dirs (also keeps them off $HOME).
export XDG_DATA_HOME="$RUN_DIR/.xdg/data"
export XDG_STATE_HOME="$RUN_DIR/.xdg/state"
export XDG_CACHE_HOME="$RUN_DIR/.xdg/cache"
mkdir -p "$XDG_DATA_HOME" "$XDG_STATE_HOME" "$XDG_CACHE_HOME"

# Per-task PyTensor compile dir. pytensor defaults to ~/.pytensor (NFS $HOME),
# shared across concurrent tasks, so its C-compile filelock contends and goes
# stale ("OSError: [Errno 116] Stale file handle"). Give each task a private
# compiledir on node-local SSD (reliable flock, fast, and off $HOME).
PYTENSOR_BASE="${L_SCRATCH:-$RUN_DIR/.cache}/pytensor"
mkdir -p "$PYTENSOR_BASE"
export PYTENSOR_FLAGS="base_compiledir=$PYTENSOR_BASE"

echo "[task $TASK] repeat=$REPEAT gt=$GT seed=$SEED -> $RUN_DIR"
[[ -f "$GT_MODELS_SRC/${GT}.py" ]] || { echo "ERROR: GT snapshot missing $GT_MODELS_SRC/${GT}.py (setup job didn't run?)" >&2; exit 1; }

# --- isolated, sanitized repo copy on scratch ------------------------------
rsync -a \
  --exclude '.git' --exclude '.venv' --exclude 'data' \
  --exclude '__pycache__' --exclude '*.nc' \
  --exclude '.uv_cache' --exclude '.pip_cache' --exclude '.cache' --exclude '.hf' \
  --exclude 'ground_truth_models.py' --exclude 'evaluate_recovery.py' \
  "$REPO"/ "$RUN_REPO"/

# pyprojroot.here() sentinel (we excluded .git from the copy).
touch "$RUN_REPO/.here"

# Physically remove THIS task's held-out GT seed-model file from the copy.
GT_FILE="$RUN_REPO/$SEED_MODELS_REL/${GT}.py"
rm -f "$GT_FILE"
[[ -e "$GT_FILE" ]] && { echo "ERROR: could not remove $GT_FILE" >&2; exit 1; }
echo "[task $TASK] removed held-out GT from copy: $SEED_MODELS_REL/${GT}.py"

# Harden opencode permissions: deny reads of the off-limits paths, and grant
# external access to the repo copy. The outer agents' cwd IS the repo copy so
# they don't need it, but inner-loop candidate agents run with cwd = their
# candidate dir and must reach the model zoo / project files above it. The GT
# seed file is deleted and seed_models/ground_truth/evaluate are read-denied
# (external_directory grants the *location*; the read deny still blocks the
# *operation*), so the grant exposes nothing sensitive.
"$VENV_PY" - "$RUN_REPO/opencode.json" "$RUN_REPO" <<'PY'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
repo = sys.argv[2]
cfg = json.loads(p.read_text()) if p.exists() else {"$schema": "https://opencode.ai/config.json"}
perm = cfg.setdefault("permission", {})
perm["read"] = {"*": "allow", "**/seed_models/**": "deny",
                "**/ground_truth_models.py": "deny", "**/evaluate_recovery.py": "deny"}
perm["glob"] = {"*": "allow", "**/seed_models/**": "deny"}
perm["grep"] = {"*": "allow", "**/seed_models/**": "deny"}
ext = perm.setdefault("external_directory", {})
ext[f"{repo}/**"] = "allow"
ext[f"{repo}/*"] = "allow"
p.write_text(json.dumps(cfg, indent=2))
print(f"[task] hardened opencode permissions: {p}")
PY

cd "$RUN_REPO"

# Optional cheap-run overrides (set by SMOKE mode or by hand). Values are
# numeric, so unquoted word-splitting below is safe.
EXTRA_FLAGS=""
[[ -n "${N_EXPERIMENTS:-}"        ]] && EXTRA_FLAGS+=" --n-experiments $N_EXPERIMENTS"
[[ -n "${N_PARTICIPANTS:-}"       ]] && EXTRA_FLAGS+=" --n-participants $N_PARTICIPANTS"
[[ -n "${INNER_LOOP_ITERATIONS:-}" ]] && EXTRA_FLAGS+=" --inner-loop-iterations $INNER_LOOP_ITERATIONS"
[[ -n "${INNER_LOOP_CANDIDATES:-}" ]] && EXTRA_FLAGS+=" --inner-loop-candidates $INNER_LOOP_CANDIDATES"
[[ -n "${DRAWS:-}"                ]] && EXTRA_FLAGS+=" --draws $DRAWS"
[[ -n "${TUNE:-}"                 ]] && EXTRA_FLAGS+=" --tune $TUNE"
[[ -n "${AGENT_TIMEOUT_SEC:-}"    ]] && EXTRA_FLAGS+=" --agent-timeout-sec $AGENT_TIMEOUT_SEC"

# Run the shared venv's interpreter directly (built by the setup job).
# --gt-model restricts this task to the single held-out model.
# --gt-models-dir reads the GT generator from the pristine scratch snapshot,
#   so the deleted copy file can't leak; seeding still excludes it.
# --chains maps onto the requested cores. --resume skips finished stages.
"$VENV_PY" scripts/subjective_randomness/holdout_recovery.py \
  --config "$CONFIG" \
  --gt-model "$GT" \
  --gt-models-dir "$GT_MODELS_SRC" \
  --seed "$SEED" \
  --backend opencode \
  --chains "${CHAINS:-4}" \
  --out      "$RUN_DIR/holdout.json" \
  --tidy-csv "$RUN_DIR/holdout.csv" \
  --figure   "$RUN_DIR/holdout.png" \
  --results-root "$RUN_REPO/_runs" \
  --cache-dir    "$RUN_DIR/mcmc_cache" \
  --resume $EXTRA_FLAGS

echo "[task $TASK] done -> $RUN_DIR/holdout.json"
