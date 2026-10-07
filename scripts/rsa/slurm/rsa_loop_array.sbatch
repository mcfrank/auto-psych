#!/bin/bash
#SBATCH --job-name=rsa_loop
#SBATCH --partition=normal
#SBATCH --array=0-5
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.out
#
# RSA run 1, step 3: one array task per cell (_cells.sh maps task -> cell:
# 3 conditions x 2 replicates). submit.sh submits it after the setup job and
# sets CPUs, memory and time from its variables (the #SBATCH values above are
# only defaults for a direct sbatch).
#
# Per cell: build the cell's agent tree, check it, run the RSA inner loop
# (5 rounds x 6 slots, NUTS 4 x 1000 + 1000, sandboxed Gemini 3.8 Flash
# agents, 2,400 s each), then score every model on the held-out conditions
# (all cells) and score recovery of the ground truth (recovery cells).
#
# LEAK PREVENTION (as scripts/subjective_randomness/slurm/holdout_recovery_array.sbatch):
#   * agents run in bubblewrap with their tree, the loop's results dir and the
#     venv read-only and only their candidate dir writable
#     (src/runtime/agent_sandbox.py); their environment is allowlisted;
#   * the tree is agent_src (agent_tree.exclude: no data, docs, tests,
#     scripts, split/simulation/scoring code, CLAUDE.md) with this cell's
#     excluded seeds' files and manifest entries removed (scrub_agent_tree.sh);
#   * the tree lives under an opaque random id ($AGENT_TREES_ROOT/<id>), never
#     under a condition-named path, and the loop's results dir is
#     <tree>/_runs/loop: every path agents are shown is opaque;
#   * the only data the agents can read is the loop's copy of this cell's
#     train.csv (<results>/responses.csv). test.csv, split.json, the combined
#     trials and the simulation's provenance.json stay in $DATA_ROOT, which
#     no sandbox mounts;
#   * check_agent_tree.sh verifies all of that before the loop starts: no
#     withheld seed file, manifest entry or byte-identical copy, no test,
#     split or provenance file, no CSV outside _runs/, no excluded path.
#
# State: $CELLS_ROOT/<cell>/ holds code_commit, cell.json, agent_tree_id,
# links repo/ and results/ (into the agent tree), last_exit, attempts/,
# outputs/ (the results without .fit_cache, copied at the end) and DONE.
# A cell is not resumable mid-loop (the loop re-seeds its zoo on start): a
# failed loop is restarted with RESTART=1, which moves the old results to
# attempts/ and keeps their fit cache. A finished loop whose scoring failed
# is resumed at the scoring by simply resubmitting the task.
set -euo pipefail
export WORK_ROOT="${WORK_ROOT:-${SCRATCH:-${GROUP_SCRATCH:?set WORK_ROOT (or SCRATCH)}}/auto-psych/rsa_run1}"
CODE_COMMIT_FILE="$WORK_ROOT/code_commit"
[[ -f "$CODE_COMMIT_FILE" && -d "$WORK_ROOT/harness_repo" && -d "$WORK_ROOT/agent_src" ]] \
  || { echo "ERROR: $WORK_ROOT has no staged code (code_commit, harness_repo, agent_src); did rsa_setup.sbatch run?" >&2; exit 1; }
STAGED_CODE=$(cat "$CODE_COMMIT_FILE")
SLURM_DIR="$WORK_ROOT/harness_repo/scripts/rsa/slurm"
SR_SLURM_DIR="$WORK_ROOT/harness_repo/scripts/subjective_randomness/slurm"
source "$SLURM_DIR/_env.sh"
rsa_require_opencode
rsa_require_gemini_key

# --- production settings (PI decisions 2026-10-07) ------------------------------
MAX_ITERATIONS="${MAX_ITERATIONS:-5}"
CANDIDATE_COUNT="${CANDIDATE_COUNT:-6}"
NUM_WARMUP="${NUM_WARMUP:-1000}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
NUM_CHAINS="${NUM_CHAINS:-4}"
AGENT_MODEL="${AGENT_MODEL:-google/gemini-3.8-flash}"
AGENT_TIMEOUT_SEC="${AGENT_TIMEOUT_SEC:-2400}"

# Agents' `uv run` goes to the tree's shim (scrub_agent_tree.sh); a stray uv
# must never be pointed at the shared venv.
unset UV_PROJECT_ENVIRONMENT UV_CACHE_DIR UV_PYTHON

# --- task -> cell -----------------------------------------------------------------
TASK="${SLURM_ARRAY_TASK_ID:?not an array task}"
rsa_cell "$TASK"
DATA_DIR="$(rsa_condition_data_dir "$DATA_ROOT" "$CONDITION")"
CELL_DIR="$CELLS_ROOT/$CELL"
mkdir -p "$CELL_DIR"
echo "[task $TASK] cell $CELL: condition=$CONDITION replicate=$REPLICATE gt=${GT:-none} excluded=[${EXCLUDED_SEEDS}] loop_seed=$LOOP_SEED"

# One job per cell; a duplicate exits before touching anything.
if ! bash "$SR_SLURM_DIR/cell_lock.sh" "$CELL_DIR/.cell_lock" "${SLURM_JOB_ID:-local}"; then
  echo "[task $TASK] $CELL_DIR is being worked on by another job; exiting without touching it."
  exit 0
fi
on_exit() {
  local status=$?
  printf '%s job=%s at=%s\n' "$status" "${SLURM_JOB_ID:-local}" "$(date -Is)" > "$CELL_DIR/last_exit"
  rm -f "$CELL_DIR/.cell_lock"
}
trap on_exit EXIT
if [[ -f "$CELL_DIR/DONE" ]]; then
  echo "[task $TASK] $CELL is complete (DONE); nothing to do."
  exit 0
fi

# The code this cell runs on: recorded at its start; never resumed on other code.
if [[ -f "$CELL_DIR/code_commit" ]]; then
  [[ "$(cat "$CELL_DIR/code_commit")" == "$STAGED_CODE" ]] \
    || { echo "ERROR: $CELL started on code $(cat "$CELL_DIR/code_commit"), the sweep's staged code is $STAGED_CODE" >&2; exit 1; }
elif [[ -n "$(ls -A "$CELL_DIR" | grep -Fvx -e .cell_lock -e last_exit)" ]]; then
  echo "ERROR: $CELL_DIR has earlier work but no record of its code; refusing to run it on $STAGED_CODE" >&2; exit 1
else
  printf '%s\n' "$STAGED_CODE" > "$CELL_DIR/code_commit"
fi

# --- the cell's data ------------------------------------------------------------
for f in train.csv test.csv split.json; do
  [[ -f "$DATA_DIR/$f" ]] || { echo "ERROR: $DATA_DIR/$f is missing (prepare_data.sh / rsa_setup.sbatch)" >&2; exit 1; }
done
GT_FILE=""
if [[ -n "$GT" ]]; then
  GT_FILE="$HARNESS_REPO/$RSA_SEED_MODELS_REL/$GT.py"
  # The simulation is from this ground truth's file and this real split.
  sha256sum --quiet -c "$DATA_DIR/DONE" \
    || { echo "ERROR: $DATA_DIR/DONE does not match the ground truth or the real train split; re-run setup" >&2; exit 1; }
  "$VENV_PY" - "$DATA_DIR/provenance.json" "$GT_FILE" <<'PY'
import hashlib, json, sys
prov, gt = json.load(open(sys.argv[1])), open(sys.argv[2], "rb").read()
if prov["ground_truth_sha256"] != hashlib.sha256(gt).hexdigest():
    sys.exit(f"the simulation's ground truth is not {sys.argv[2]}")
PY
fi
if [[ ! -f "$CELL_DIR/cell.json" ]]; then
  "$VENV_PY" - "$CELL_DIR/cell.json" <<PY
import json, sys
json.dump(dict(
    cell="$CELL", condition="$CONDITION", replicate=$REPLICATE, task=$TASK,
    ground_truth="$GT" or None, excluded_seeds="$EXCLUDED_SEEDS".split(),
    loop_seed=$LOOP_SEED, data_dir="$DATA_DIR", code="$STAGED_CODE",
    max_iterations=$MAX_ITERATIONS, candidate_count=$CANDIDATE_COUNT,
    num_warmup=$NUM_WARMUP, num_samples=$NUM_SAMPLES, num_chains=$NUM_CHAINS,
    agent_model="$AGENT_MODEL", agent_timeout_sec=$AGENT_TIMEOUT_SEC,
), open(sys.argv[1], "w"), indent=1)
PY
  (cd "$DATA_DIR" && sha256sum train.csv test.csv split.json) > "$CELL_DIR/data.sha256"
fi
(cd "$DATA_DIR" && sha256sum --quiet -c "$CELL_DIR/data.sha256") \
  || { echo "ERROR: $CELL's data changed since it started ($CELL_DIR/data.sha256)" >&2; exit 1; }

# --- the agent tree, under an opaque id ---------------------------------------------
if [[ ! -s "$CELL_DIR/agent_tree_id" ]]; then
  head -c 8 /dev/urandom | od -An -tx1 | tr -d ' \n' > "$CELL_DIR/agent_tree_id"
fi
AGENT_DIR="$AGENT_TREES_ROOT/$(cat "$CELL_DIR/agent_tree_id")"
RUN_REPO="$AGENT_DIR/repo"
RESULTS="$RUN_REPO/_runs/loop"
for word in ${GT:+$CONDITION $GT}; do  # a recovery cell's path must not name it
  [[ "$AGENT_DIR" != *"$word"* ]] || { echo "ERROR: the agent directory $AGENT_DIR names '$word' (set AGENT_TREES_ROOT elsewhere)" >&2; exit 1; }
done
mkdir -p "$RUN_REPO"
ln -sfn "$RUN_REPO" "$CELL_DIR/repo"
ln -sfn "$RESULTS" "$CELL_DIR/results"

# Per-task opencode/XDG state and PyTensor compile dir (as the SR array).
export XDG_DATA_HOME="$AGENT_DIR/.xdg/data"
export XDG_STATE_HOME="$AGENT_DIR/.xdg/state"
export XDG_CACHE_HOME="$AGENT_DIR/.xdg/cache"
mkdir -p "$XDG_DATA_HOME" "$XDG_STATE_HOME" "$XDG_CACHE_HOME"
PYTENSOR_BASE="${L_SCRATCH:-$AGENT_DIR/.cache}/pytensor/job_${SLURM_JOB_ID:-local}"
mkdir -p "$PYTENSOR_BASE"
export PYTENSOR_FLAGS="base_compiledir=$PYTENSOR_BASE"

# The harness and the agents reach the venv through an opaque link: the
# sandbox mounts it at the path the harness runs it by.
VENV_DIR="$(dirname "$(dirname "$VENV_PY")")"
[[ -f "$VENV_DIR/pyvenv.cfg" ]] || { echo "ERROR: $VENV_PY is not inside a venv" >&2; exit 1; }
ln -sfn "$VENV_DIR" "$AGENT_DIR/venv"
AGENT_PY="$AGENT_DIR/venv/bin/python"

# shellcheck disable=SC2086  # EXCLUDED_SEEDS is a list of plain names
bash "$SLURM_DIR/build_agent_tree.sh" "$AGENT_SRC" "$RUN_REPO" "$AGENT_PY" $EXCLUDED_SEEDS

CHECK_ARGS=()
for seed in $EXCLUDED_SEEDS; do CHECK_ARGS+=(--gt-file "$HARNESS_REPO/$RSA_SEED_MODELS_REL/$seed.py"); done
while IFS= read -r -d '' f; do CHECK_ARGS+=(--forbid-file "$f"); done < <(
  find "$DATA_ROOT" -maxdepth 2 -type f \( -name test.csv -o -name split.json -o -name provenance.json \
       -o -name combined_trials.csv -o -name simulated.csv \) -print0)
bash "$SLURM_DIR/check_agent_tree.sh" "$RUN_REPO" ${CHECK_ARGS[@]+"${CHECK_ARGS[@]}"}

# --- the loop ---------------------------------------------------------------------
cd "$HARNESS_REPO"
SEED_ARGS=()
if rsa_loop_cli_has_seed "$AGENT_PY"; then
  SEED_ARGS=(--seed "$LOOP_SEED")
elif [[ "${ALLOW_SHARED_LOOP_SEED:-}" == 1 ]]; then
  echo "[task $TASK] WARNING: the loop CLI has no --seed (ALLOW_SHARED_LOOP_SEED=1): this replicate differs from the other only by the agents' sampling"
  echo "shared (the loop CLI had no --seed)" > "$CELL_DIR/loop_seed_note"
else
  echo "ERROR: the loop CLI has no --seed; set ALLOW_SHARED_LOOP_SEED=1 to run replicates that share every seed" >&2; exit 1
fi
EXCLUDE_ARGS=()
if [[ -n "$EXCLUDED_SEEDS" ]]; then
  read -r -a seeds <<< "$EXCLUDED_SEEDS"
  EXCLUDE_ARGS=(--exclude-seeds "${seeds[@]}")
fi

if [[ -f "$RESULTS/export.json" ]]; then
  echo "[task $TASK] the loop finished in an earlier job ($RESULTS/export.json); going on to scoring"
else
  RESUME_ARGS=()
  if [[ -d "$RESULTS" && -n "$(ls -A "$RESULTS")" && "${RESTART:-}" != 1 && -f "$RESULTS/history.json" ]]; then
    # An interrupted loop: continue from its last scored step (src.rsa.loop
    # --resume abandons a half-finished round and runs it again).
    RESUME_ARGS=(--resume)
    echo "[task $TASK] resuming the unfinished loop in $RESULTS from its last scored step"
  elif [[ -d "$RESULTS" && -n "$(ls -A "$RESULTS")" ]]; then
    if [[ "${RESTART:-}" != 1 ]]; then
      echo "ERROR: $RESULTS holds a loop that stopped before its seeds were scored (no history.json)." >&2
      echo "       Resubmit this task with RESTART=1 to move it to $CELL_DIR/attempts/ and start over (its fit cache is kept)." >&2
      exit 1
    fi
    mkdir -p "$CELL_DIR/attempts"
    attempt="$CELL_DIR/attempts/attempt_$(date +%Y%m%dT%H%M%S)"
    mv "$RESULTS" "$attempt"
    mkdir -p "$RESULTS"
    # Fits are content-addressed (model, data, settings): reusing them is exact.
    [[ -d "$attempt/.fit_cache" ]] && mv "$attempt/.fit_cache" "$RESULTS/.fit_cache"
    echo "[task $TASK] RESTART=1: moved the unfinished loop to $attempt (kept its .fit_cache)"
  fi

  # The agents inherit the harness's environment through the sandbox's
  # allowlist; withhold anyway every variable that points into WORK_ROOT or
  # the checkout or (in a recovery cell) names the condition or ground truth.
  WITHHELD=()
  for name in $(compgen -e); do
    value="${!name}"
    if [[ "$value" == *"$WORK_ROOT"* || "$value" == *"$REPO"* \
          || ( -n "$GT" && ( "$value" == *"$GT"* || "$value" == *"$CONDITION"* ) ) ]]; then
      WITHHELD+=("$name")
    fi
  done
  UNSET_ARGS=()
  for name in ${WITHHELD[@]+"${WITHHELD[@]}"}; do UNSET_ARGS+=(-u "$name"); done
  echo "[task $TASK] withheld from the agents' environment: ${WITHHELD[*]:-}"

  echo "[task $TASK] loop: ${MAX_ITERATIONS} rounds x ${CANDIDATE_COUNT} slots, NUTS ${NUM_CHAINS} x (${NUM_WARMUP} + ${NUM_SAMPLES}), $AGENT_MODEL, ${AGENT_TIMEOUT_SEC} s/agent -> $RESULTS"
  env ${UNSET_ARGS[@]+"${UNSET_ARGS[@]}"} PATH="$RUN_REPO/.agent_bin:$PATH" \
    "$AGENT_PY" -m src.rsa.loop.run \
      --results "$RESULTS" \
      --responses "$DATA_DIR/train.csv" \
      ${EXCLUDE_ARGS[@]+"${EXCLUDE_ARGS[@]}"} \
      --max-iterations "$MAX_ITERATIONS" --candidate-count "$CANDIDATE_COUNT" \
      --num-warmup "$NUM_WARMUP" --num-samples "$NUM_SAMPLES" --num-chains "$NUM_CHAINS" \
      --agent-root "$RUN_REPO" \
      --agent-model "$AGENT_MODEL" --agent-timeout-sec "$AGENT_TIMEOUT_SEC" \
      ${SEED_ARGS[@]+"${SEED_ARGS[@]}"} \
      ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
  [[ -f "$RESULTS/export.json" ]] || { echo "ERROR: the loop exited 0 without $RESULTS/export.json" >&2; exit 1; }

  # Warn-only records of what the agents did: did any write the ground
  # truth's name (it may be a coincidence), and what did they look at.
  if [[ -n "$GT" ]]; then
    grep -rIl -- "$GT" "$RESULTS"/round_* > "$CELL_DIR/gt_name_mentions.txt" 2>/dev/null || true
    if [[ -s "$CELL_DIR/gt_name_mentions.txt" ]]; then
      echo "[task $TASK] WARNING: agent output names the ground truth '$GT' in $(wc -l < "$CELL_DIR/gt_name_mentions.txt") file(s): $CELL_DIR/gt_name_mentions.txt"
    else
      rm -f "$CELL_DIR/gt_name_mentions.txt"
    fi
  fi
  "$VENV_PY" "$SR_SLURM_DIR/agent_activity_report.py" --agent-dir "$AGENT_DIR" \
    --out "$CELL_DIR/agent_activity.md" --expected "$HOME" "$VENV_DIR" \
    || echo "[task $TASK] WARNING: the agent activity report failed (warn-only)"
fi

# --- scoring (no agents run from here on) -------------------------------------------
# The loop's NUTS settings and seed, so scoring reads the loop's own fits from
# its cache (refits included) instead of fitting every model again.
NUTS_ARGS=(--num-warmup "$NUM_WARMUP" --num-samples "$NUM_SAMPLES" --num-chains "$NUM_CHAINS" --seed "$LOOP_SEED")
if [[ ! -f "$RESULTS/heldout/heldout.json" ]]; then
  echo "[task $TASK] held-out evaluation on $DATA_DIR/test.csv"
  "$VENV_PY" -m src.rsa.evaluate_heldout --loop-dir "$RESULTS" --test "$DATA_DIR/test.csv" \
    --out-dir "$RESULTS/heldout" "${NUTS_ARGS[@]}"
fi
if [[ -n "$GT" && ! -f "$RESULTS/recovery/recovery.json" ]]; then
  echo "[task $TASK] recovery scoring against $GT"
  "$VENV_PY" -m src.rsa.recovery --loop-dir "$RESULTS" --gt "$GT_FILE" --test "$DATA_DIR/test.csv" \
    --out-dir "$RESULTS/recovery" "${NUTS_ARGS[@]}"
fi

# --- outputs: everything but the fit cache, outside the agent tree -------------------
rsync -a --delete --exclude '/.fit_cache/' --exclude '.home/' "$RESULTS"/ "$CELL_DIR/outputs"/
date -Is > "$CELL_DIR/DONE"
echo "[task $TASK] $CELL done: $CELL_DIR/outputs (agent tree kept: $RUN_REPO)"
