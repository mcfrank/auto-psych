#!/bin/bash
#SBATCH --job-name=impossible_holdout_recovery
#SBATCH --partition=normal
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.out
#
# Stage 2/3: one array task per (repeat, impossible ground-truth model). With R
# repeats and G ground truths the submit script sizes the array to R*G; task T
# maps to
#   repeat = (T-1)/G + 1   (distinct --seed per repeat -> different synthetic data)
#   gt     = GT_MODELS[(T-1)%G]
#
# LEAK PREVENTION — impossible variant. The agent gets the FULL, normal seed pool
# (unlike the standard holdout, nothing is held out of the seed set: the ground
# truth is not a seed model). What must NOT reach the agent is the impossible
# recipe, which leaks two ways:
#   * the impossible model FILES (e.g. more_heads_more_random.py) — the literal
#     generator; and
#   * their giveaway NAMES in the config (more_heads_more_random, ...) — which
#     announce the generating rule.
# So each task:
#   * runs from its OWN rsync copy of the repo on $SCRATCH that EXCLUDES the
#     impossible_models/ directory and the impossible config file, so neither the
#     recipe nor the answer-bearing names sit on the agent's disk;
#   * feeds the parent a per-task config (written OUTSIDE the agent's cwd) whose
#     gt_models_dir is rewritten to the pristine snapshot the setup job staged
#     ($IMPOSSIBLE_MODELS_SRC), so the parent can still load the GT while the copy
#     can't;
#   * additionally denies opencode read/glob/grep on impossible_models/ and the
#     config.
# The experiment tree (results-root) lives INSIDE the agent's cwd (the repo copy)
# so opencode needs no external-directory grant for it — exactly like the normal
# pipeline. The MCMC cache + holdout.{json,csv,png} live at $WORK_ROOT/run<r>/<gt>/.

set -euo pipefail
SLURM_DIR="${HOLDOUT_SLURM_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "$SLURM_DIR/_env.sh"

# The coding agents inherit this shell's environment. _env.sh exports
# UV_PROJECT_ENVIRONMENT pointing at the SHARED per-run venv; agents invoke
# `uv run python ...` to test their work, and each such call re-syncs that venv
# against the full project lock — wiping our installed packages mid-run. The
# pipeline uses $VENV_PY (an explicit interpreter path), so drop the UV_* vars
# here: a stray agent `uv run` then targets a throwaway .venv in its own repo
# copy, never the shared environment.
unset UV_PROJECT_ENVIRONMENT UV_CACHE_DIR UV_PYTHON

# --- task -> (repeat, ground-truth model) ----------------------------------
BASE_SEED="${BASE_SEED:-0}"
read -r -a GTS <<< "${GT_MODELS:-more_heads_more_random fewer_heads_more_random longer_runs_more_random more_imbalance_more_random}"
N_GTS="${N_GTS:-${#GTS[@]}}"
CONFIG="${CONFIG:-scripts/subjective_randomness/configs/impossible_holdout_recovery.yaml}"
IMPOSSIBLE_MODELS_REL="${IMPOSSIBLE_MODELS_REL:-src/subjective_randomness/impossible_models}"
IMPOSSIBLE_MODELS_SRC="${IMPOSSIBLE_MODELS_SRC:-$WORK_ROOT/impossible_models_src}"   # pristine recipe snapshot
IMPOSSIBLE_CONFIG_SRC="${IMPOSSIBLE_CONFIG_SRC:-$WORK_ROOT/impossible_config_src.yaml}"
CONFIG_BASENAME="$(basename "$CONFIG")"

TASK="${SLURM_ARRAY_TASK_ID:-1}"
REPEAT=$(( (TASK - 1) / N_GTS + 1 ))
GT="${GTS[$(( (TASK - 1) % N_GTS ))]}"
SEED=$(( BASE_SEED + REPEAT ))

RUN_DIR="$WORK_ROOT/run${REPEAT}/${GT}"
RUN_REPO="$RUN_DIR/repo"
mkdir -p "$RUN_DIR"

# Per-task opencode state: opencode keeps its session SQLite DB under
# XDG_DATA_HOME. Concurrent array tasks sharing one DB corrupt it. Isolate every
# task's XDG dirs (also keeps them off $HOME).
export XDG_DATA_HOME="$RUN_DIR/.xdg/data"
export XDG_STATE_HOME="$RUN_DIR/.xdg/state"
export XDG_CACHE_HOME="$RUN_DIR/.xdg/cache"
mkdir -p "$XDG_DATA_HOME" "$XDG_STATE_HOME" "$XDG_CACHE_HOME"

# Per-task PyTensor compile dir on node-local SSD (reliable flock, fast, off
# $HOME — the ~/.pytensor default contends and goes stale across tasks).
PYTENSOR_BASE="${L_SCRATCH:-$RUN_DIR/.cache}/pytensor"
mkdir -p "$PYTENSOR_BASE"
export PYTENSOR_FLAGS="base_compiledir=$PYTENSOR_BASE"

echo "[task $TASK] repeat=$REPEAT gt=$GT seed=$SEED -> $RUN_DIR"
[[ -f "$IMPOSSIBLE_MODELS_SRC/${GT}.py" ]] || { echo "ERROR: impossible snapshot missing $IMPOSSIBLE_MODELS_SRC/${GT}.py (setup job didn't run?)" >&2; exit 1; }
[[ -f "$IMPOSSIBLE_CONFIG_SRC" ]] || { echo "ERROR: staged config missing $IMPOSSIBLE_CONFIG_SRC (setup job didn't run?)" >&2; exit 1; }

# --- isolated, sanitized repo copy on scratch ------------------------------
# Excludes carry over from the standard pipeline (.git/.venv/data/caches/
# ground_truth/evaluate), PLUS the impossible recipe and config so neither lands
# on the agent's disk.
rsync -a \
  --exclude '.git' --exclude '.venv' --exclude 'data' \
  --exclude '__pycache__' --exclude '*.nc' \
  --exclude '.uv_cache' --exclude '.pip_cache' --exclude '.cache' --exclude '.hf' \
  --exclude 'ground_truth_models.py' --exclude 'evaluate_recovery.py' \
  --exclude 'impossible_models' --exclude "$CONFIG_BASENAME" \
  "$REPO"/ "$RUN_REPO"/

# pyprojroot.here() sentinel (we excluded .git from the copy).
touch "$RUN_REPO/.here"

# Prove the recipe and the answer-bearing config NAMES are absent from the copy.
LEAKED_DIR="$RUN_REPO/$IMPOSSIBLE_MODELS_REL"
LEAKED_CFG="$RUN_REPO/$(dirname "$CONFIG")/$CONFIG_BASENAME"
[[ -e "$LEAKED_DIR" ]] && { echo "ERROR: impossible recipe leaked into copy: $LEAKED_DIR" >&2; exit 1; }
[[ -e "$LEAKED_CFG" ]] && { echo "ERROR: impossible config leaked into copy: $LEAKED_CFG" >&2; exit 1; }
echo "[task $TASK] copy is clean: no impossible_models/, no $CONFIG_BASENAME"

# Per-task config OUTSIDE the agent's cwd: same as the staged config but with
# gt_models_dir rewritten to the pristine snapshot (absolute -> resolve_path
# returns it unchanged, so the parent loads the GT from there, off the copy).
TASK_CONFIG="$RUN_DIR/config.yaml"
"$VENV_PY" - "$IMPOSSIBLE_CONFIG_SRC" "$IMPOSSIBLE_MODELS_SRC" "$TASK_CONFIG" <<'PY'
import sys, pathlib, yaml
src, models_src, dst = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = yaml.safe_load(pathlib.Path(src).read_text())
cfg["gt_models_dir"] = models_src
pathlib.Path(dst).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(f"[task] wrote sanitized config -> {dst} (gt_models_dir={models_src})")
PY

# Harden opencode permissions: deny reads of the off-limits paths, and grant
# external access to the repo copy. The outer agents' cwd IS the repo copy so
# they don't need it, but inner-loop candidate agents run with cwd = their
# candidate dir and must reach the model zoo / project files above it. The
# impossible recipe/config are absent from the copy AND read-denied here
# (defense in depth); seed_models/ground_truth/evaluate stay denied as in the
# standard pipeline.
"$VENV_PY" - "$RUN_REPO/opencode.json" "$RUN_REPO" <<'PY'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
repo = sys.argv[2]
cfg = json.loads(p.read_text()) if p.exists() else {"$schema": "https://opencode.ai/config.json"}
perm = cfg.setdefault("permission", {})
perm["read"] = {"*": "allow", "**/seed_models/**": "deny",
                "**/impossible_models/**": "deny",
                "**/impossible_holdout_recovery.yaml": "deny",
                "**/ground_truth_models.py": "deny", "**/evaluate_recovery.py": "deny"}
perm["glob"] = {"*": "allow", "**/seed_models/**": "deny",
                "**/impossible_models/**": "deny"}
perm["grep"] = {"*": "allow", "**/seed_models/**": "deny",
                "**/impossible_models/**": "deny"}
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
# --gt-model restricts this task to the single held-out impossible model.
# --config points at the per-task config whose gt_models_dir is the pristine
#   snapshot, so the deleted copy files can't leak.
# --chains maps onto the requested cores. --resume skips finished stages.
"$VENV_PY" scripts/subjective_randomness/impossible_holdout_recovery.py \
  --config "$TASK_CONFIG" \
  --gt-model "$GT" \
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
