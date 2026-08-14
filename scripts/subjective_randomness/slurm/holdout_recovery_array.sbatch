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
#   * the copy EXCLUDES ground_truth_models.py / evaluate_recovery.py, the
#     literature-model test files, and gt.txt (the source paper the faithful
#     models implement), and has THIS task's held-out model deleted from BOTH
#     PyMC locations (the registry $SEED_MODELS_REL and the live seed pool), so
#     the GT recipe is not on the agent's disk;
#   * the ONE remaining on-disk copy of the GT recipe is the pure-Python family
#     twin (src/subjective_randomness/model_families/<gt>.py) — the parent
#     imports it for DEFAULT_PARAMS so it cannot be deleted; opencode read/
#     glob/grep denies cover it (and both model-family dirs wholesale);
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
read -r -a GTS <<< "${GT_MODELS:-falk_konold_dp motif_stack finite_experience_occurrence local_representativeness}"
N_GTS="${N_GTS:-${#GTS[@]}}"
# The GT/baseline registry (config seed_models_dir) — where the GT's PyMC file
# is deleted from the copy and where the pristine snapshot was staged from.
SEED_MODELS_REL="${SEED_MODELS_REL:-src/subjective_randomness/pymc_model_families}"
# The live project seed pool experiment 1 is seeded from. Since the 2026-08
# consolidation it mirrors the registry, so an active-seed GT also has a copy
# here that must go.
POOL_MODELS_REL="${POOL_MODELS_REL:-src/pipelines/outer_loop/projects/subjective_randomness/seed_models}"
CONFIG="${CONFIG:-scripts/subjective_randomness/configs/holdout_recovery_faithful.yaml}"
GT_MODELS_SRC="${GT_MODELS_SRC:-$WORK_ROOT/gt_models_src}"  # pristine GT snapshot
GT_FAMILY_REL="${GT_FAMILY_REL:-src/subjective_randomness/model_families}"
GT_FAMILY_SRC="${GT_FAMILY_SRC:-$WORK_ROOT/gt_family_src}"  # pristine family snapshot

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
# Besides caches/VCS, exclude every GT-recipe side channel the parent never
# reads: the recovery-harness sources, the literature-model test files (they
# encode each faithful model's behavior), and gt.txt (the extracted source
# paper the faithful models implement).
rsync -a \
  --exclude '.git' --exclude '.venv' --exclude 'data' \
  --exclude '__pycache__' --exclude '*.nc' \
  --exclude '.uv_cache' --exclude '.pip_cache' --exclude '.cache' --exclude '.hf' \
  --exclude 'ground_truth_models.py' --exclude 'evaluate_recovery.py' \
  --exclude 'test_literature_faithful_pymc.py' \
  --exclude 'test_literature_model_families.py' \
  --exclude 'gt.txt' --exclude 'REFACTOR_NOTES.md' \
  "$REPO"/ "$RUN_REPO"/

# pyprojroot.here() sentinel (we excluded .git from the copy).
touch "$RUN_REPO/.here"

# Physically remove THIS task's held-out GT PyMC file from the copy — from the
# registry AND from the live seed pool (which mirrors the registry since the
# 2026-08 consolidation; under a superseded-GT config the pool copy simply
# doesn't exist).
for gt_copy in "$RUN_REPO/$SEED_MODELS_REL/${GT}.py" "$RUN_REPO/$POOL_MODELS_REL/${GT}.py"; do
  rm -f "$gt_copy"
  if [[ -e "$gt_copy" ]]; then echo "ERROR: could not remove $gt_copy" >&2; exit 1; fi
done
echo "[task $TASK] removed held-out GT from copy: $SEED_MODELS_REL/${GT}.py + $POOL_MODELS_REL/${GT}.py"

# Scrub the pure-Python family twin too (the real GT-recipe leak — an agent
# running its own Python could read its DEFAULT_PARAMS + generative form, which
# is how a prior run pasted the GT's delta/alpha into a candidate). Replace it
# with an importable stub carrying no params or code: the package __init__
# imports every twin, so the file must stay importable, but the parent reads the
# real DEFAULT_PARAMS from the pristine $GT_FAMILY_SRC via --gt-family-dir (by
# parsing, not importing). Opencode denies below remain as defence in depth.
[[ -f "$GT_FAMILY_SRC/${GT}.py" ]] || { echo "ERROR: pristine family snapshot missing $GT_FAMILY_SRC/${GT}.py (setup job didn't stage it?)" >&2; exit 1; }
GT_FAMILY_COPY="$RUN_REPO/$GT_FAMILY_REL/${GT}.py"
if [[ -f "$GT_FAMILY_COPY" ]]; then
  cat > "$GT_FAMILY_COPY" <<'STUB'
"""Held-out ground truth. Its real DEFAULT_PARAMS and generative form live in
the pristine gt_family snapshot, off every agent's disk. This importable stub
exists only so ``model_families/__init__`` (which imports every twin) still
loads; the held-out family is never fit or called by the pipeline."""

DEFAULT_PARAMS: dict = {}


def _held_out(*_args, **_kwargs):
    raise RuntimeError("held-out ground-truth family is unavailable to the agent")


score_sequence = predict_left = predict = _held_out
STUB
  echo "[task $TASK] scrubbed held-out GT family twin -> stub: $GT_FAMILY_REL/${GT}.py"
fi

# Harden opencode permissions: deny reads of the off-limits paths, and grant
# external access to the repo copy. The outer agents' cwd IS the repo copy so
# they don't need it, but inner-loop candidate agents run with cwd = their
# candidate dir and must reach the model zoo / project files above it. The GT
# PyMC files are deleted; the denies additionally cover both model-family dirs
# (the pure-Python twin of the GT must stay importable for the parent), the
# seed pool, and the holdout configs (their gt_models keys name the held-out
# model). external_directory grants the *location*; the read deny still blocks
# the *operation*, so the grant exposes nothing sensitive.
"$VENV_PY" - "$RUN_REPO/opencode.json" "$RUN_REPO" <<'PY'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
repo = sys.argv[2]
cfg = json.loads(p.read_text()) if p.exists() else {"$schema": "https://opencode.ai/config.json"}
perm = cfg.setdefault("permission", {})
deny_globs = [
    "**/seed_models/**",
    "**/model_families/**",
    "**/pymc_model_families/**",
    "**/configs/holdout_recovery*.yaml",
    "**/ground_truth_models.py",
    "**/evaluate_recovery.py",
    "**/gt.txt",
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
  --gt-family-dir "$GT_FAMILY_SRC" \
  --seed "$SEED" \
  --backend opencode \
  --chains "${CHAINS:-4}" \
  --out      "$RUN_DIR/holdout.json" \
  --tidy-csv "$RUN_DIR/holdout.csv" \
  --figure   "$RUN_DIR/holdout.png" \
  --results-root "$RUN_REPO/_runs" \
  --summary-root "$WORK_ROOT/run${REPEAT}" \
  --cache-dir    "$RUN_DIR/mcmc_cache" \
  ${DESIGN_N_EIG:+--design-n-eig $DESIGN_N_EIG} \
  ${DESIGN_N_RANDOM:+--design-n-random $DESIGN_N_RANDOM} \
  --resume $EXTRA_FLAGS

echo "[task $TASK] done -> $RUN_DIR/holdout.json"
