#!/bin/bash
#SBATCH --job-name=rsa_outer_rehearsal
#SBATCH --partition=mcfrank
#SBATCH --array=0-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=48:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.out
#
# The simulated dress rehearsal of the live campaign (PI 2026-10-09): one run of
# three experiments per selection scope, with real agents (sandboxed, no
# network) and simulated participants answering from a hidden ground truth.
#   task 0: the inner loop selects on all data so far (cumulative)
#   task 1: it fits on all data but selects on the live trials only
# Same ground truth, seed and seeds in both: experiment 1's design and
# simulated people are identical, so the two runs differ only by how the
# inner loop selects. After promote.sbatch (it reuses run 2's promote_work).
#
#   mkdir -p "$SCRATCH/auto-psych/rsa_rehearsal/logs"
#   sbatch --chdir="$HOME/auto-psych" -o "$SCRATCH/auto-psych/rsa_rehearsal/logs/%x_%A_%a.out" \
#     scripts/rsa/slurm/outer_rehearsal.sbatch
#
# The ground truth is picked by rule (src.rsa.outer.ground_truth: the starting
# model farthest from every promoted seed) and withheld from the agents' tree.
# Everything that names it (its fit, the configuration, prospective and
# recovery records) stays in the cell's private/, outside the tree.
# Overrides: PARTICIPANTS (200), DISPLAYS (20), N_EXPERIMENTS (3), AGENT_MODEL,
# AGENT_TIMEOUT_SEC, SEED (0), RUN2_ROOT (run 2's sweep root).
# Resubmitting a task resumes its run where it stopped.
set -euo pipefail
export REPO="${REPO:-${SLURM_SUBMIT_DIR:-$HOME/auto-psych}}"
export WORK_ROOT="${WORK_ROOT:-${SCRATCH:?}/auto-psych/rsa_rehearsal}"
source "$REPO/scripts/rsa/slurm/_env.sh"
rsa_require_opencode
rsa_require_gemini_key
command -v bwrap >/dev/null || { echo "FATAL: no bwrap (sandboxed agents need it)" >&2; exit 1; }
command -v flock >/dev/null || { echo "FATAL: no flock" >&2; exit 1; }

TASK="${SLURM_ARRAY_TASK_ID:?run as an array task (0: all, 1: live)}"
SCOPES=(all live)
SCOPE="${SCOPES[$TASK]:?task $TASK is not 0 or 1}"
RUN2_ROOT="${RUN2_ROOT:-$(rsa_default_work_root)}"  # the run-2 sweep (_cells.sh)
PROMOTE_WORK="$RUN2_ROOT/promote_work"
EXISTING="$PROMOTE_WORK/all_trials.csv"
AGENT_MODEL="${AGENT_MODEL:-google/gemini-3.8-flash}"
AGENT_TIMEOUT_SEC="${AGENT_TIMEOUT_SEC:-2400}"
SEED="${SEED:-0}"
for f in "$EXISTING" "$REPO/data/rsa/live_seeds/models/models_manifest.yaml"; do
  [[ -f "$f" ]] || { echo "ERROR: $f is missing: run promote.sbatch and commit/pull its output first" >&2; exit 1; }
done
mkdir -p "$WORK_ROOT" "$CELLS_ROOT"

# --- once per rehearsal: staged code, the seeds, the ground truth ------------------
(
  flock 9
  bash "$REPO/scripts/rsa/slurm/stage_code.sh"
  if [[ ! -d "$WORK_ROOT/live_seeds" ]]; then
    cp -r "$REPO/data/rsa/live_seeds/models" "$WORK_ROOT/live_seeds.tmp" && mv "$WORK_ROOT/live_seeds.tmp" "$WORK_ROOT/live_seeds"
  fi
  if [[ ! -f "$WORK_ROOT/ground_truth.json" ]]; then
    (cd "$WORK_ROOT/harness_repo" && "$VENV_PY" -m src.rsa.outer.ground_truth --promoted "$WORK_ROOT/live_seeds" \
       --data "$EXISTING" --cache "$PROMOTE_WORK/.fit_cache" --out "$WORK_ROOT/ground_truth.json")
  fi
) 9>"$WORK_ROOT/.setup.lock"
HARNESS_REPO="$WORK_ROOT/harness_repo"
SLURM_DIR="$HARNESS_REPO/scripts/rsa/slurm"
GT="$("$VENV_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["ground_truth"])' "$WORK_ROOT/ground_truth.json")"
GT_FILE="$HARNESS_REPO/$RSA_SEED_MODELS_REL/$GT.py"
[[ -f "$GT_FILE" ]] || { echo "ERROR: no ground-truth file $GT_FILE" >&2; exit 1; }

# --- the cell and its agent tree (as rsa_loop_array.sbatch) --------------------------
CELL="rehearsal_$SCOPE"
CELL_DIR="$CELLS_ROOT/$CELL"
mkdir -p "$CELL_DIR/private"
[[ -s "$CELL_DIR/agent_tree_id" ]] || head -c 8 /dev/urandom | od -An -tx1 | tr -d ' \n' > "$CELL_DIR/agent_tree_id"
AGENT_DIR="$AGENT_TREES_ROOT/$(cat "$CELL_DIR/agent_tree_id")"
RUN_REPO="$AGENT_DIR/repo"
RUN_DIR="$RUN_REPO/_runs/outer"
[[ "$AGENT_DIR" != *"$GT"* ]] || { echo "ERROR: the agent directory names the ground truth" >&2; exit 1; }
mkdir -p "$RUN_REPO"
export XDG_DATA_HOME="$AGENT_DIR/.xdg/data" XDG_STATE_HOME="$AGENT_DIR/.xdg/state" XDG_CACHE_HOME="$AGENT_DIR/.xdg/cache"
mkdir -p "$XDG_DATA_HOME" "$XDG_STATE_HOME" "$XDG_CACHE_HOME"
PYTENSOR_BASE="${L_SCRATCH:-$AGENT_DIR/.cache}/pytensor/job_${SLURM_JOB_ID:-local}"
mkdir -p "$PYTENSOR_BASE"
export PYTENSOR_FLAGS="base_compiledir=$PYTENSOR_BASE"
VENV_DIR="$(dirname "$(dirname "$VENV_PY")")"
ln -sfn "$VENV_DIR" "$AGENT_DIR/venv"
AGENT_PY="$AGENT_DIR/venv/bin/python"
bash "$SLURM_DIR/build_agent_tree.sh" "$WORK_ROOT/agent_src" "$RUN_REPO" "$AGENT_PY" "$GT"
bash "$SLURM_DIR/check_agent_tree.sh" "$RUN_REPO" --gt-file "$GT_FILE"

# --- the run ------------------------------------------------------------------------
WITHHELD=()
for name in $(compgen -e); do
  value="${!name}"
  if [[ "$value" == *"$WORK_ROOT"* || "$value" == *"$REPO"* || "$value" == *"$RUN2_ROOT"* || "$value" == *"$GT"* ]]; then
    WITHHELD+=("$name")
  fi
done
UNSET_ARGS=()
for name in ${WITHHELD[@]+"${WITHHELD[@]}"}; do UNSET_ARGS+=(-u "$name"); done
echo "[rehearsal $SCOPE] withheld from the agents' environment: ${WITHHELD[*]:-}"
echo "[rehearsal $SCOPE] ${N_EXPERIMENTS:-3} experiments, ${PARTICIPANTS:-200} people x 10 displays of ${DISPLAYS:-20}, $AGENT_MODEL -> $RUN_DIR"
cd "$HARNESS_REPO"
env ${UNSET_ARGS[@]+"${UNSET_ARGS[@]}"} PATH="$RUN_REPO/.agent_bin:$PATH" \
  "$AGENT_PY" -m src.rsa.outer.run --run-dir "$RUN_DIR" --private-dir "$CELL_DIR/private" \
    --seeds "$WORK_ROOT/live_seeds" --existing-data "$EXISTING" --collection simulated --ground-truth "$GT_FILE" \
    --selection-scope "$SCOPE" --n-experiments "${N_EXPERIMENTS:-3}" --participants "${PARTICIPANTS:-200}" \
    --trials 10 --displays "${DISPLAYS:-20}" --seed "$SEED" \
    --agent-root "$RUN_REPO" --agent-model "$AGENT_MODEL" --agent-timeout-sec "$AGENT_TIMEOUT_SEC"

# --- outputs: the run without fit caches, and the private records --------------------
rsync -a --delete --exclude '.fit_cache/' --exclude '.home/' --exclude '.agent_shell/' "$RUN_DIR"/ "$CELL_DIR/outputs"/
grep -rIlw -- "$GT" "$RUN_DIR"/experiment*/model_loop/round_* > "$CELL_DIR/gt_name_mentions.txt" 2>/dev/null || true
date -Is > "$CELL_DIR/DONE"
echo "[rehearsal $SCOPE] done: $CELL_DIR/outputs and $CELL_DIR/private"
