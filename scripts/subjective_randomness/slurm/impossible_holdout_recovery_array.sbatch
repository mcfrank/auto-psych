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
# recipe: the model files (e.g. more_heads_more_random.py) and their giveaway
# NAMES, which announce the generating rule. The agent tree is built exactly as
# the faithful array builds it (holdout_recovery_array.sbatch):
#   * from the sweep's staged agent source ($WORK_ROOT/agent_src, scrubbed with
#     agent_tree.exclude, which drops src/subjective_randomness/ with the
#     impossible models and scripts/ with every config);
#   * under an opaque, random id in $AGENT_TREES_ROOT, never under the GT-named
#     cell directory (run<r>/<gt>/repo is a symlink to it). Until September 2026
#     this array kept the tree at run<r>/<gt>/repo, so every agent prompt named
#     the recipe, and built it with an old inline exclude list that kept tests/,
#     docs/, CLAUDE.md and .secrets;
#   * the harness runs from the staged harness_repo with --agent-root, and
#     reads the recipe from the pristine snapshot the setup job staged;
#   * the agents' environment carries no variable naming the model or leading to
#     the full source, and the sandbox gives each agent only an allowlist;
#   * the GT-name scans run before the agents (fatal) and after (warn-only).

set -euo pipefail
# Every cell runs on the code the setup job staged once for the sweep.
: "${WORK_ROOT:?WORK_ROOT is not set}"
HARNESS_REPO="${HARNESS_REPO:-$WORK_ROOT/harness_repo}"
AGENT_SRC="${AGENT_SRC:-$WORK_ROOT/agent_src}"
CODE_COMMIT_FILE="$WORK_ROOT/code_commit"
[[ -f "$CODE_COMMIT_FILE" && -d "$HARNESS_REPO" && -d "$AGENT_SRC" ]] \
  || { echo "ERROR: $WORK_ROOT has no staged code (code_commit, harness_repo, agent_src); the setup job didn't run?" >&2; exit 1; }
STAGED_CODE=$(cat "$CODE_COMMIT_FILE")
SLURM_DIR="$HARNESS_REPO/scripts/subjective_randomness/slurm"
source "$SLURM_DIR/_env.sh"

# The coding agents inherit this shell's environment; a stray agent `uv run`
# must not re-sync the shared venv (see holdout_recovery_array.sbatch).
unset UV_PROJECT_ENVIRONMENT UV_CACHE_DIR UV_PYTHON

# --- task -> (repeat, ground-truth model) ----------------------------------
BASE_SEED="${BASE_SEED:-0}"
read -r -a GTS <<< "${GT_MODELS:-more_heads_more_random fewer_heads_more_random longer_runs_more_random more_imbalance_more_random}"
N_GTS="${N_GTS:-${#GTS[@]}}"
IMPOSSIBLE_MODELS_SRC="${IMPOSSIBLE_MODELS_SRC:-$WORK_ROOT/impossible_models_src}"   # pristine recipe snapshot
IMPOSSIBLE_CONFIG_SRC="${IMPOSSIBLE_CONFIG_SRC:-$WORK_ROOT/impossible_config_src.yaml}"

TASK="${SLURM_ARRAY_TASK_ID:-1}"
REPEAT=$(( (TASK - 1) / N_GTS + 1 ))
GT="${GTS[$(( (TASK - 1) % N_GTS ))]}"
SEED=$(( BASE_SEED + REPEAT ))

RUN_DIR="$WORK_ROOT/run${REPEAT}/${GT}"
mkdir -p "$RUN_DIR"
# One job per cell (cell_lock.sh); the lock goes when this job exits.
if ! bash "$SLURM_DIR/cell_lock.sh" "$RUN_DIR/.cell_lock" "${SLURM_JOB_ID:-local}"; then
  echo "[task $TASK] $RUN_DIR is being worked on by another job; exiting without touching it."
  exit 0
fi
trap 'rm -f "$RUN_DIR/.cell_lock"' EXIT
# A finished cell is left alone (its archive and leak record included).
if [[ -f "$RUN_DIR/holdout.json" ]]; then
  echo "[task $TASK] $RUN_DIR is already complete (holdout.json); nothing to do."
  exit 0
fi
# The code this cell runs on: recorded when it starts; never resumed on other code.
if [[ -f "$RUN_DIR/code_commit" ]]; then
  cell_code=$(cat "$RUN_DIR/code_commit")
  if [[ "$cell_code" != "$STAGED_CODE" ]]; then
    echo "ERROR: $RUN_DIR started on code $cell_code, but this sweep's staged code is $STAGED_CODE; refusing to resume it on other code." >&2
    exit 1
  fi
elif [[ -n "$(ls -A "$RUN_DIR" | grep -Fvx '.cell_lock')" ]]; then
  echo "ERROR: $RUN_DIR has earlier work but no record of the code it ran on; refusing to resume it on $STAGED_CODE." >&2
  exit 1
else
  printf '%s\n' "$STAGED_CODE" > "$RUN_DIR/code_commit"
fi
echo "[task $TASK] code: $STAGED_CODE"

# The agents' tree lives under an opaque, random id recorded in the cell, not
# under the GT-named RUN_DIR: every path in it reaches an agent's prompt.
AGENT_TREES_ROOT="${AGENT_TREES_ROOT:-${SCRATCH:-$GROUP_SCRATCH}/auto-psych/agent_trees}"
if [[ ! -s "$RUN_DIR/agent_tree_id" ]]; then
  head -c 8 /dev/urandom | od -An -tx1 | tr -d ' \n' > "$RUN_DIR/agent_tree_id"
fi
AGENT_DIR="$AGENT_TREES_ROOT/$(cat "$RUN_DIR/agent_tree_id")"
RUN_REPO="$AGENT_DIR/repo"
[[ "$AGENT_DIR" != *"$GT"* ]] || { echo "ERROR: the agent directory names the held-out model: $AGENT_DIR (set AGENT_TREES_ROOT elsewhere)" >&2; exit 1; }
mkdir -p "$RUN_REPO"
if [[ -e "$RUN_DIR/repo" && ! -L "$RUN_DIR/repo" ]]; then
  echo "ERROR: $RUN_DIR/repo is a real directory, from a task started before the agent tree moved out of the cell directory. Resume it with the script it started under, or use a new WORK_ROOT." >&2
  exit 1
fi
ln -sfn "$RUN_REPO" "$RUN_DIR/repo"

# Per-task opencode state and PyTensor compile dir (see
# holdout_recovery_array.sbatch), under the opaque AGENT_DIR.
export XDG_DATA_HOME="$AGENT_DIR/.xdg/data"
export XDG_STATE_HOME="$AGENT_DIR/.xdg/state"
export XDG_CACHE_HOME="$AGENT_DIR/.xdg/cache"
mkdir -p "$XDG_DATA_HOME" "$XDG_STATE_HOME" "$XDG_CACHE_HOME"
PYTENSOR_BASE="${L_SCRATCH:-$AGENT_DIR/.cache}/pytensor"
mkdir -p "$PYTENSOR_BASE"
export PYTENSOR_FLAGS="base_compiledir=$PYTENSOR_BASE"

echo "[task $TASK] repeat=$REPEAT gt=$GT seed=$SEED -> $RUN_DIR"
[[ -f "$IMPOSSIBLE_MODELS_SRC/${GT}.py" ]] || { echo "ERROR: impossible snapshot missing $IMPOSSIBLE_MODELS_SRC/${GT}.py (setup job didn't run?)" >&2; exit 1; }
[[ -f "$IMPOSSIBLE_CONFIG_SRC" ]] || { echo "ERROR: staged config missing $IMPOSSIBLE_CONFIG_SRC (setup job didn't run?)" >&2; exit 1; }

# --- isolated, sanitized agent tree on scratch --------------------------------
AGENT_EXCLUDE="$SLURM_DIR/agent_tree.exclude"
[[ -f "$AGENT_EXCLUDE" ]] || { echo "ERROR: $AGENT_EXCLUDE not found" >&2; exit 1; }
rsync -a --delete --delete-excluded --filter='P /_runs/***' \
  --exclude-from="$AGENT_EXCLUDE" "$AGENT_SRC"/ "$RUN_REPO"/
touch "$RUN_REPO/.here"  # pyprojroot sentinel (.git is not copied)

# Harden opencode permissions: deny reads of the off-limits paths (defence in
# depth: they are not in the tree) and grant external access to the repo copy.
"$VENV_PY" - "$RUN_REPO/opencode.json" "$RUN_REPO" <<'PY'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
repo = sys.argv[2]
cfg = json.loads(p.read_text()) if p.exists() else {"$schema": "https://opencode.ai/config.json"}
perm = cfg.setdefault("permission", {})
deny_globs = [
    "**/seed_models/**",
    "**/impossible_models/**",
    "**/model_families/**",
    "**/pymc_model_families/**",
    "**/configs/*holdout_recovery*.yaml",
    "**/ground_truth_models.py",
    "**/evaluate_recovery.py",
]
perm["read"] = {"*": "allow", **{g: "deny" for g in deny_globs}}
perm["glob"] = {"*": "allow", **{g: "deny" for g in deny_globs}}
perm["grep"] = {"*": "allow", **{g: "deny" for g in deny_globs}}
ext = perm.setdefault("external_directory", {})
ext[f"{repo}/**"] = "allow"
ext[f"{repo}/*"] = "allow"
p.write_text(json.dumps(cfg, indent=2))
print(f"[task] hardened opencode permissions: {p}")
PY

# Prove the recipe is absent from the tree, and that no file names it.
for forbidden_dir in src/subjective_randomness scripts; do
  if [[ -d "$RUN_REPO/$forbidden_dir" ]]; then
    echo "ERROR: $forbidden_dir is present in the agent tree; it holds the impossible recipe or configs naming it. Check agent_tree.exclude." >&2
    exit 1
  fi
done
bash "$SLURM_DIR/scan_gt_name.sh" --before-agents "$GT" "$RUN_REPO" /dev/null
echo "[task $TASK] GT-name scan: '$GT' appears nowhere in the agent tree"

# Per-task config OUTSIDE the agent tree: the staged config with gt_models_dir
# rewritten to the pristine snapshot, so the parent loads the GT from there.
TASK_CONFIG="$RUN_DIR/config.yaml"
"$VENV_PY" - "$IMPOSSIBLE_CONFIG_SRC" "$IMPOSSIBLE_MODELS_SRC" "$TASK_CONFIG" <<'PY'
import sys, pathlib, yaml
src, models_src, dst = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = yaml.safe_load(pathlib.Path(src).read_text())
cfg["gt_models_dir"] = models_src
pathlib.Path(dst).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(f"[task] wrote sanitized config -> {dst} (gt_models_dir={models_src})")
PY

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

# The harness runs from the full staged copy; agents run from the scrubbed tree.
cd "$HARNESS_REPO"

# Two more paths reach agents' briefs (the MCMC cache, the interpreter): hand
# the harness both through opaque links.
mkdir -p "$RUN_DIR/mcmc_cache"
ln -sfn "$RUN_DIR/mcmc_cache" "$AGENT_DIR/mcmc_cache"
VENV_DIR="$(dirname "$(dirname "$VENV_PY")")"
[[ -f "$VENV_DIR/pyvenv.cfg" ]] || { echo "ERROR: $VENV_PY is not inside a venv (no $VENV_DIR/pyvenv.cfg)" >&2; exit 1; }
ln -sfn "$VENV_DIR" "$AGENT_DIR/venv"

# Withhold from the harness's environment (which the agents' allowlist is taken
# from) every variable that names the model or points into WORK_ROOT or REPO.
WITHHELD=()
for name in $(compgen -e); do
  value="${!name}"
  if [[ "$name" == GT_MODELS || "$name" == CONFIG || "$value" == *"$GT"* \
        || "$value" == *"$WORK_ROOT"* || "$value" == *"$REPO"* ]]; then
    WITHHELD+=("$name")
  fi
done
UNSET_ARGS=()
for name in "${WITHHELD[@]}"; do UNSET_ARGS+=(-u "$name"); done
echo "[task $TASK] withheld from the agents' environment: ${WITHHELD[*]}"

env "${UNSET_ARGS[@]}" "$AGENT_DIR/venv/bin/python" scripts/subjective_randomness/impossible_holdout_recovery.py \
  --config "$TASK_CONFIG" \
  --gt-model "$GT" \
  --seed "$SEED" \
  --backend "${AGENT_BACKEND:-opencode}" \
  ${AGENT_MODEL:+--agent-model "$AGENT_MODEL"} \
  --chains "${CHAINS:-4}" \
  --out      "$RUN_DIR/holdout.json" \
  --tidy-csv "$RUN_DIR/holdout.csv" \
  --figure   "$RUN_DIR/holdout.png" \
  --results-root "$RUN_REPO/_runs" \
  --summary-root "$WORK_ROOT/run${REPEAT}" \
  --cache-dir    "$AGENT_DIR/mcmc_cache" \
  --agent-root   "$RUN_REPO" \
  --resume $EXTRA_FLAGS

echo "[task $TASK] done -> $RUN_DIR/holdout.json"

# Did any agent write the recipe's name? Warn-only (gt_name_mentions.txt).
bash "$SLURM_DIR/scan_gt_name.sh" --after-run "$GT" "$RUN_REPO/_runs" "$RUN_DIR/gt_name_mentions.txt"
"$VENV_PY" "$SLURM_DIR/agent_activity_report.py" --agent-dir "$AGENT_DIR" \
  --out "$RUN_DIR/agent_activity.md" --expected "$HOME" "$(dirname "$(dirname "$VENV_PY")")" \
  || echo "[task $TASK] WARNING: the agent activity report failed (warn-only; the cell is unaffected)"

# --- reclaim inodes: archive + remove this task's agent tree on success -------
if [[ -z "${KEEP_REPO_COPY:-}" ]]; then
  # Removes the tree only once the archive reads back; never overwrites an
  # existing archive (archive_agent_tree.sh).
  bash "$SLURM_DIR/archive_agent_tree.sh" "$TASK" "$RUN_DIR" "$AGENT_DIR"
else
  echo "[task $TASK] KEEP_REPO_COPY set — leaving repo copy in place: $RUN_REPO"
fi
