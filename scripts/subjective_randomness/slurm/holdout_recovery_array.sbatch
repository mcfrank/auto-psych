#!/bin/bash
#SBATCH --job-name=holdout_recovery
#SBATCH --partition=normal
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.out
#
# CPUs and memory: the inner loop fits models concurrently (fit_models_cached
# sizes its process pool so workers x cores-per-fit fills the CPUs Slurm
# allocates), so 16 cores run four 4-chain fits at once where 8 ran one with
# half the allocation idle. Memory scales with the concurrent fits: the
# September 2026 sweep peaked at 13.8-18.7 GB per task (sacct MaxRSS) fitting
# one model at a time under --mem=32GB, so 64GB leaves the same margin for
# four. `normal` allows at most 8 GB/core (sh_part), i.e. 128 GB at 16 cores.
# If a future sweep needs more, `-p hns` (the lab-owned partition) allows up to
# 256 cores per node, 26 GB/core and 7 days.
#
# Stage 2/3: one array task per (repeat, ground-truth model). With R repeats and
# G ground truths the submit script sizes the array to R*G; task T maps to
#   repeat = (T-1)/G + 1   (distinct --seed per repeat -> different synthetic data)
#   gt     = GT_MODELS[(T-1)%G]
#
# LEAK PREVENTION. The agents run as this Unix user with no read sandbox
# (claude skips permissions; codex's sandbox is pinned open for reads; opencode
# agents read through bash). An agent that goes LOOKING can find the ground
# truth — `ps` shows this script's arguments. What this script guarantees is
# that nothing hands it to them without looking:
#   * each task's agents work in their OWN scrubbed copy of the repo, built with
#     agent_tree.exclude: no feature code, no model-family implementations
#     (model_families/ and pymc_model_families/), no GT-recipe or evaluation
#     code, no holdout configs, no CLAUDE.md/AGENTS.md (agent CLIs load those
#     into every session);
#   * the held-out model's registry file, pool file and manifest entries are
#     removed, and so is every file named after it (its tests and analysis
#     scripts);
#   * no path an agent is shown names the held-out model. The copy lives under
#     an opaque id in $AGENT_TREES_ROOT, not in the GT-named cell directory
#     (run<r>/<gt>/repo is a symlink to it for the harness-side tools), and the
#     harness names its run directory cell_<i>. Until September 2026 every
#     agent prompt carried "run1/motif_stack/.../_runs/motif_stack/";
#   * the agents' environment carries no variable naming the model or leading
#     to the full source (GT_MODELS, CONFIG, WORK_ROOT, REPO, ...);
#   * the parent reads the GT generator from pristine snapshots the setup job
#     staged outside every agent tree ($WORK_ROOT/gt_models_src, gt_family_src).
# A pre-flight check enforces the file and path guarantees before any agent
# starts. holdout.{json,csv,png} and the MCMC cache live at $WORK_ROOT/run<r>/<gt>/.

set -euo pipefail
# Every cell of a sweep, first attempt or retry, runs on the code the setup job
# staged once for the sweep: the harness repo (which also holds these Slurm
# scripts) and the agent source. Nothing below reads the live checkout's code.
: "${WORK_ROOT:?WORK_ROOT is not set}"
HARNESS_REPO="${HARNESS_REPO:-$WORK_ROOT/harness_repo}"
AGENT_SRC="${AGENT_SRC:-$WORK_ROOT/agent_src}"
CODE_COMMIT_FILE="$WORK_ROOT/code_commit"
[[ -f "$CODE_COMMIT_FILE" && -d "$HARNESS_REPO" && -d "$AGENT_SRC" ]] \
  || { echo "ERROR: $WORK_ROOT has no staged code (code_commit, harness_repo, agent_src); the setup job didn't run?" >&2; exit 1; }
STAGED_CODE=$(cat "$CODE_COMMIT_FILE")
SLURM_DIR="$HARNESS_REPO/scripts/subjective_randomness/slurm"
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
mkdir -p "$RUN_DIR"
# One job per cell: the retry job and the sweep watcher can both resubmit a
# failed cell, and two jobs in one cell would corrupt it. A duplicate exits
# here, before touching anything; the lock goes when this job exits.
if ! bash "$SLURM_DIR/cell_lock.sh" "$RUN_DIR/.cell_lock" "${SLURM_JOB_ID:-local}"; then
  echo "[task $TASK] $RUN_DIR is being worked on by another job; exiting without touching it."
  exit 0
fi
trap 'rm -f "$RUN_DIR/.cell_lock"' EXIT
# A finished cell is left alone. Slurm can mark a task that finished as
# OUT_OF_MEMORY or TIMEOUT (in its final clean-up), and a resubmission that ran
# on would rebuild an empty _runs/, overwrite the real agent_runs.tar.gz with
# it, and blank the cell's leak record (gt_name_mentions.txt, agent_activity.md).
if [[ -f "$RUN_DIR/holdout.json" ]]; then
  echo "[task $TASK] $RUN_DIR is already complete (holdout.json); nothing to do."
  exit 0
fi
# The code this cell runs on: recorded when the cell starts, and a cell is
# never resumed on other code. A cell directory holding anything but its lock
# and no record started before records were kept, on unknown code.
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

# The agents' tree must not live under the GT-named RUN_DIR: every path in it
# reaches an agent's prompt. It lives under an opaque, random id, recorded in
# RUN_DIR/agent_tree_id so a resubmitted task finds its tree again, and
# RUN_DIR/repo links to it. (The id used to be a hash of RUN_DIR: an agent
# that found $WORK_ROOT, e.g. in the venv's bin/activate, could hash the four
# candidate paths and confirm the ground truth.)
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

# Per-task opencode state: opencode keeps its session SQLite DB under
# XDG_DATA_HOME (default ~/.local/share/opencode). Concurrent array tasks
# sharing one DB corrupt it ("SQLITE_CORRUPT: database disk image is
# malformed"). Isolate every task's XDG dirs (also keeps them off $HOME).
# Within a task, the launcher gives each spawned agent its own XDG_DATA_HOME
# under the agent's directory (concurrent candidate agents hit "database is
# locked" on one shared store); this one serves the harness process itself.
# These are exported, so every agent inherits them: they live under the
# opaque AGENT_DIR, not the GT-named RUN_DIR.
export XDG_DATA_HOME="$AGENT_DIR/.xdg/data"
export XDG_STATE_HOME="$AGENT_DIR/.xdg/state"
export XDG_CACHE_HOME="$AGENT_DIR/.xdg/cache"
mkdir -p "$XDG_DATA_HOME" "$XDG_STATE_HOME" "$XDG_CACHE_HOME"

# Per-task PyTensor compile dir. pytensor defaults to ~/.pytensor (NFS $HOME),
# shared across concurrent tasks, so its C-compile filelock contends and goes
# stale ("OSError: [Errno 116] Stale file handle"). Give each task a private
# compiledir on node-local SSD (reliable flock, fast, and off $HOME).
PYTENSOR_BASE="${L_SCRATCH:-$AGENT_DIR/.cache}/pytensor"
mkdir -p "$PYTENSOR_BASE"
export PYTENSOR_FLAGS="base_compiledir=$PYTENSOR_BASE"

echo "[task $TASK] repeat=$REPEAT gt=$GT seed=$SEED -> $RUN_DIR"
[[ -f "$GT_MODELS_SRC/${GT}.py" ]] || { echo "ERROR: GT snapshot missing $GT_MODELS_SRC/${GT}.py (setup job didn't run?)" >&2; exit 1; }

# --- isolated, sanitized agent tree on scratch --------------------------------
# The agent tree is the scrubbed repo copy the coding agents run from. It is
# built with the shared exclude file (agent_tree.exclude) which removes feature
# code, research library modules, GT-recipe files, holdout configs, and build
# artefacts. The HARNESS runs from a full copy (harness_repo, staged by the
# setup job) so it can import everything it needs; the agents see only the
# scrubbed tree. node_modules exclusion alone saves ~5.5k inodes per copy.
AGENT_EXCLUDE="$SLURM_DIR/agent_tree.exclude"
[[ -f "$AGENT_EXCLUDE" ]] || { echo "ERROR: $AGENT_EXCLUDE not found" >&2; exit 1; }
# --delete-excluded also cleans a tree a resumed task built before an exclusion
# was added (resumed cells kept docs/, tests/ and .secrets otherwise); the
# protect filter keeps the agents' results root, whose data/ and *.nc match
# exclude patterns.
rsync -a --delete --delete-excluded --filter='P /_runs/***' \
  --exclude-from="$AGENT_EXCLUDE" "$AGENT_SRC"/ "$RUN_REPO"/

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
# external access to the repo copy. Every agent runs with cwd = the repo copy
# and the launcher (src/runtime/coding_agent.py) pins PWD to that cwd, so
# opencode roots its session there, loads THIS opencode.json, and treats the
# whole tree (including _runs/, the results root) as internal — the grant is
# belt-and-braces. (Before P34 the harness's `cd "$HARNESS_REPO"` below leaked
# into the agents' PWD, opencode rooted every session at harness_repo instead,
# and the agents' own candidate dirs were "external" and auto-rejected.) The GT
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

# Remove the held-out model's ENTRY from both manifests the agents can open.
# Deleting the .py above leaves its manifest line behind, and that line is the
# model's name plus a one-sentence rationale stating its mechanism — i.e. the
# answer. Rewriting the manifest also drops the header comment that lists every
# active model. Seeding keys off the pool manifest by name, so a name that is
# simply absent is seeded around exactly like a superseded GT (--missing-ok
# covers that case: a superseded GT was never listed). Run from $RUN_REPO so
# pyprojroot resolves the copy; the --models-dir paths are absolute regardless.
for models_rel in "$SEED_MODELS_REL" "$POOL_MODELS_REL"; do
  # A directory the agent tree excludes outright (agent_tree.exclude now drops
  # the whole PyMC registry, whose files describe the ground truths) has nothing
  # left to scrub. An absent directory is fine; a present one without a manifest
  # is still an error.
  if [[ ! -d "$RUN_REPO/$models_rel" ]]; then
    echo "[task $TASK] $models_rel is not in the agent tree (excluded) — nothing to scrub"
    continue
  fi
  [[ -f "$RUN_REPO/$models_rel/models_manifest.yaml" ]] || { echo "ERROR: no manifest at $RUN_REPO/$models_rel" >&2; exit 1; }
  "$VENV_PY" "$HARNESS_REPO/scripts/subjective_randomness/remove_manifest_entry.py" \
    --models-dir "$RUN_REPO/$models_rel" --name "$GT" --missing-ok
  if grep -q "$GT" "$RUN_REPO/$models_rel/models_manifest.yaml"; then
    echo "ERROR: $models_rel/models_manifest.yaml still mentions $GT" >&2; exit 1
  fi
done
echo "[task $TASK] scrubbed held-out GT entry from both manifests"

# Catch-all: scrub the GT name from ANY remaining models_manifest.yaml in the
# checkout. The loop above handles the active feature regime's pair, but the
# other regime's manifests also exist and the leakage audit scans all of them.
while IFS= read -r -d '' mf; do
  if grep -q "$GT" "$mf"; then
    "$VENV_PY" "$HARNESS_REPO/scripts/subjective_randomness/remove_manifest_entry.py" \
      --models-dir "$(dirname "$mf")" --name "$GT" --missing-ok
    if grep -q "$GT" "$mf"; then
      echo "ERROR: $mf still mentions $GT after scrub" >&2; exit 1
    fi
    mf_rel="${mf#$RUN_REPO/}"
    echo "[task $TASK] scrubbed $GT from additional manifest: $mf_rel"
  fi
done < <(find "$RUN_REPO" -name "models_manifest.yaml" -not -path "*/_runs/*" -print0)

# --- GT-leak verifier: backend-agnostic, runs for EVERY backend --------------
# Everything above this point that protects the ground truth is either physical
# (rsync excludes, rm of <gt>.py, manifest scrubs) or a permission rule in
# opencode.json. Only opencode reads that file: the claude backend runs with
# --dangerously-skip-permissions, and codex's sandbox is pinned by enterprise
# policy to a profile with filesystem read access. So for those backends the
# deny-list is inert and the physical scrub is the ONLY protection. Assert it
# actually worked, before a single agent starts, rather than discovering after
# the fact that a cell "recovered" the ground truth by reading it.
# Remove every file in the agent tree NAMED after the held-out model. Beyond
# the registry entry the step above deletes, the checkout carries unit tests and
# analysis scripts for the ground truths (test_motif_stack_softmax.py,
# compare_motif_stack_predictions.py, ...). A test for a mechanism encodes that
# mechanism, and opencode's deny-list covers neither tests/ nor scripts/ — so
# these have been readable in every run to date, under every backend. Keyed on
# $GT so it generalises to any held-out model; nothing the pipeline runs lives
# under these names (the harness runs from its own full checkout).
while IFS= read -r -d '' named; do
  rm -f "$named"
  echo "[task $TASK] removed GT-named file from agent tree: ${named#$RUN_REPO/}"
done < <(find "$RUN_REPO" -name "*${GT}*" -type f -not -path "*/_runs/*" -print0)

# Assert the physical protections actually landed: the two model-family
# directories must be absent from the tree entirely, and no implementation file
# for the held-out model may remain anywhere. (The name scan below then checks
# that no file mentions it at all; docs/, tests/ and scripts/, which did, are
# no longer copied.)
for forbidden_dir in src/subjective_randomness/pymc_model_families \
                     src/subjective_randomness/model_families; do
  if [[ -d "$RUN_REPO/$forbidden_dir" ]]; then
    echo "ERROR: $forbidden_dir is present in the agent tree; it holds implementations of the ground truths." >&2
    echo "Add it to agent_tree.exclude. (Only opencode honours the read deny-list; claude and codex ignore it.)" >&2
    exit 1
  fi
done
leak_files=$(find "$RUN_REPO" -name "*${GT}*.py" -not -path "*/_runs/*" 2>/dev/null || true)
if [[ -n "$leak_files" ]]; then
  echo "ERROR: implementation files for the held-out ground truth '$GT' remain in the agent tree:" >&2
  printf '  %s\n' ${leak_files//$RUN_REPO\//} >&2
  exit 1
fi
echo "[task $TASK] GT-leak verifier: no ${GT} implementation and no model-family dir in the agent tree"
# And no file in the tree names the held-out model at all (a loader comment
# naming motif_stack reached three agents). A hit is our leak, so the cell stops
# here, before any agent has run.
bash "$SLURM_DIR/scan_gt_name.sh" --before-agents "$GT" "$RUN_REPO" /dev/null
echo "[task $TASK] GT-name scan: '$GT' appears nowhere in the agent tree"

# Optional cheap-run overrides (set by SMOKE mode or by hand). Values are
# numeric, so unquoted word-splitting below is safe.
EXTRA_FLAGS=""
[[ -n "${N_EXPERIMENTS:-}"        ]] && EXTRA_FLAGS+=" --n-experiments $N_EXPERIMENTS"
[[ -n "${N_PARTICIPANTS:-}"       ]] && EXTRA_FLAGS+=" --n-participants $N_PARTICIPANTS"
[[ -n "${INNER_LOOP_ITERATIONS:-}" ]] && EXTRA_FLAGS+=" --inner-loop-iterations $INNER_LOOP_ITERATIONS"
[[ -n "${INNER_LOOP_CANDIDATES:-}" ]] && EXTRA_FLAGS+=" --inner-loop-candidates $INNER_LOOP_CANDIDATES"
[[ -n "${NOVELTY_RMSE_THRESHOLD:-}" ]] && EXTRA_FLAGS+=" --novelty-rmse-threshold $NOVELTY_RMSE_THRESHOLD"
[[ -n "${N_CRITIQUE_PROPOSALS:-}" ]] && EXTRA_FLAGS+=" --n-critique-proposals $N_CRITIQUE_PROPOSALS"
[[ -n "${DRAWS:-}"                ]] && EXTRA_FLAGS+=" --draws $DRAWS"
[[ -n "${TUNE:-}"                 ]] && EXTRA_FLAGS+=" --tune $TUNE"
[[ -n "${AGENT_TIMEOUT_SEC:-}"    ]] && EXTRA_FLAGS+=" --agent-timeout-sec $AGENT_TIMEOUT_SEC"

# The harness runs from the FULL repo copy (harness_repo, staged once by the
# setup job), not from the scrubbed agent tree. It can therefore import
# stimulus_design, model_families, etc. that the agents must not see.
# --agent-root points agent subprocesses at the scrubbed $RUN_REPO.
cd "$HARNESS_REPO"

# Two more paths reach agents' briefs: the critique brief prints the MCMC cache
# path, the candidate brief prints the harness's interpreter (sys.executable)
# for its self-test command. Hand the harness both through opaque links.
mkdir -p "$RUN_DIR/mcmc_cache"
ln -sfn "$RUN_DIR/mcmc_cache" "$AGENT_DIR/mcmc_cache"
VENV_DIR="$(dirname "$(dirname "$VENV_PY")")"
[[ -f "$VENV_DIR/pyvenv.cfg" ]] || { echo "ERROR: $VENV_PY is not inside a venv (no $VENV_DIR/pyvenv.cfg)" >&2; exit 1; }
ln -sfn "$VENV_DIR" "$AGENT_DIR/venv"

# The agents inherit the harness's environment. Withhold every variable that
# names the held-out model or points into WORK_ROOT or REPO — both hold full
# copies of the GT code (harness_repo/, gt_models_src/, the source tree) — plus
# the GT list and the config path, whose names can describe the experiment.
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

env "${UNSET_ARGS[@]}" "$AGENT_DIR/venv/bin/python" scripts/subjective_randomness/holdout_recovery.py \
  --config "$CONFIG" \
  --gt-model "$GT" \
  --gt-models-dir "$GT_MODELS_SRC" \
  --gt-family-dir "$GT_FAMILY_SRC" \
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
  ${DESIGN_N_EIG:+--design-n-eig $DESIGN_N_EIG} \
  ${DESIGN_N_RANDOM:+--design-n-random $DESIGN_N_RANDOM} \
  --resume $EXTRA_FLAGS

echo "[task $TASK] done -> $RUN_DIR/holdout.json"

# Did any agent write the held-out model's name? It may be a coincidence, so
# this only warns (listing the files in $RUN_DIR/gt_name_mentions.txt, which the
# analysis job reports); it never fails a finished cell.
bash "$SLURM_DIR/scan_gt_name.sh" --after-run "$GT" "$RUN_REPO/_runs" "$RUN_DIR/gt_name_mentions.txt"
# What the agents looked at (URLs; paths outside their own directory), for a
# person to skim. Warn-only. Inside the sandbox $HOME is each agent's private
# home, and the Python install is mounted read-only under it.
"$VENV_PY" "$SLURM_DIR/agent_activity_report.py" --agent-dir "$AGENT_DIR" \
  --out "$RUN_DIR/agent_activity.md" --expected "$HOME" "$(dirname "$(dirname "$VENV_PY")")" \
  || echo "[task $TASK] WARNING: the agent activity report failed (warn-only; the cell is unaffected)"

# --- reclaim inodes: archive + remove this task's repo copy on success -------
# Reached only when the run above exits 0 (set -e), i.e. this GT is complete:
# holdout.{json,csv,png} are in $RUN_DIR and trajectory.json in the summary root,
# both OUTSIDE the copy. The repo copy (sanitized source + the agent's _runs
# tree) is thousands of small files and the biggest inode consumer per task; it
# is not read by the analysis job (it globs run<r>/<gt>/holdout.csv under
# $RUN_DIR, never the copy) nor by --resume (which skips finished GTs via
# trajectory.json). Collapse the agent artifacts to one tarball for later
# inspection, then delete the tree so a long sweep can't fill the inode quota.
# A FAILED task exits before here (set -e), leaving its copy intact for
# debugging and resume. Set KEEP_REPO_COPY=1 to opt out.
if [[ -z "${KEEP_REPO_COPY:-}" ]]; then
  if [[ -e "$RUN_DIR/agent_runs.tar.gz" ]]; then
    # Never overwrite an archive: it may be the only record of an earlier run.
    echo "[task $TASK] WARNING: $RUN_DIR/agent_runs.tar.gz already exists; not overwriting it, and keeping the agent tree $AGENT_DIR"
  else
    if [[ -d "$RUN_REPO/_runs" ]]; then
      tar czf "$RUN_DIR/agent_runs.tar.gz" -C "$RUN_REPO" _runs \
        && echo "[task $TASK] archived agent _runs -> $RUN_DIR/agent_runs.tar.gz"
    fi
    # rm -rf unlinks the venv and mcmc_cache symlinks inside AGENT_DIR without
    # following them: the shared venv and this cell's cache are untouched.
    rm -rf "$AGENT_DIR"
    rm -f "$RUN_DIR/repo"
    echo "[task $TASK] removed the agent tree to reclaim inodes: $AGENT_DIR"
  fi
else
  echo "[task $TASK] KEEP_REPO_COPY set — leaving repo copy in place: $RUN_REPO"
fi
