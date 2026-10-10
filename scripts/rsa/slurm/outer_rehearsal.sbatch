#!/bin/bash
#SBATCH --job-name=rsa_outer_rehearsal
#SBATCH --partition=mcfrank
#SBATCH --array=0
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=48:00:00
#SBATCH --output=%x_%A_%a.out
#
# A simulated dress rehearsal of the live campaign: one run per selection scope
# (array task i runs SCOPES[i]), with real agents (sandboxed, no network) and
# simulated participants answering from a hidden ground truth. Every run starts
# from one chain's seeds (CHAIN, default 0, of data/rsa/live_seeds/chains), as
# a live chain will. After promote.sbatch and chains.sbatch (it reuses run 2's
# promote_work).
#
# Rehearsal 2 (PI 2026-10-10, the default): one run, the guarded rule, two
# experiments, and a ground truth coherent with the human data (GT_RULE
# coherent: run 2's unpromoted model near the CV front farthest from the
# chain; src.rsa.outer.ground_truth). Rehearsal 1 (2026-10-09) was
# SCOPES="all live", GT_RULE=farthest_starting, N_EXPERIMENTS=3, in
# $SCRATCH/auto-psych/rsa_rehearsal; its literal-listener ground truth
# contradicted the human data (docs/auto_rsa/REHEARSAL_REVIEW.md).
#
#   mkdir -p "$SCRATCH/auto-psych/rsa_rehearsal2/logs"
#   sbatch --chdir="$HOME/auto-psych" -o "$SCRATCH/auto-psych/rsa_rehearsal2/logs/%x_%A_%a.out" \
#     scripts/rsa/slurm/outer_rehearsal.sbatch
#
# The ground truth is withheld from the agents' tree. Everything that names it
# (its fit, the configuration, prospective and recovery records) stays in the
# cell's private/, outside the tree.
# Overrides: SCOPES ("guarded"; with more, --array=0-<n-1>), GT_RULE (coherent),
# PARTICIPANTS (200), DISPLAYS (40), CHAIN (0), N_EXPERIMENTS (2), AGENT_MODEL,
# AGENT_TIMEOUT_SEC, SEED (0), RUN2_ROOT (run 2's sweep root).
# Resubmitting a task resumes its run where it stopped.
set -euo pipefail
export REPO="${REPO:-${SLURM_SUBMIT_DIR:-$HOME/auto-psych}}"
export WORK_ROOT="${WORK_ROOT:-${SCRATCH:?}/auto-psych/rsa_rehearsal2}"
source "$REPO/scripts/rsa/slurm/_env.sh"
rsa_require_opencode
rsa_require_gemini_key
command -v bwrap >/dev/null || { echo "FATAL: no bwrap (sandboxed agents need it)" >&2; exit 1; }
command -v flock >/dev/null || { echo "FATAL: no flock" >&2; exit 1; }

TASK="${SLURM_ARRAY_TASK_ID:?run as an array task}"
read -r -a SCOPE_LIST <<< "${SCOPES:-guarded}"
SCOPE="${SCOPE_LIST[$TASK]:?array task $TASK has no scope in SCOPES=${SCOPES:-guarded}}"
GT_RULE="${GT_RULE:-coherent}"
N_EXPERIMENTS="${N_EXPERIMENTS:-2}"
RUN2_ROOT="${RUN2_ROOT:-$(rsa_default_work_root)}"  # the run-2 sweep (_cells.sh)
PROMOTE_WORK="$RUN2_ROOT/promote_work"
EXISTING="$PROMOTE_WORK/all_trials.csv"
AGENT_MODEL="${AGENT_MODEL:-google/gemini-3.8-flash}"
AGENT_TIMEOUT_SEC="${AGENT_TIMEOUT_SEC:-2400}"
SEED="${SEED:-0}"
CHAIN="${CHAIN:-0}"
PROMOTED="$REPO/data/rsa/live_seeds/models"
CHAIN_SEEDS="$REPO/data/rsa/live_seeds/chains/chain_$CHAIN"
for f in "$EXISTING" "$PROMOTED/models_manifest.yaml" "$CHAIN_SEEDS/models_manifest.yaml"; do
  [[ -f "$f" ]] || { echo "ERROR: $f is missing: run promote.sbatch and chains.sbatch and commit/pull their output first" >&2; exit 1; }
done
mkdir -p "$WORK_ROOT" "$CELLS_ROOT"

# --- once per rehearsal: staged code, the seeds, the ground truth ------------------
(
  flock 9
  bash "$REPO/scripts/rsa/slurm/stage_code.sh"
  if [[ ! -d "$WORK_ROOT/live_seeds" ]]; then
    cp -r "$CHAIN_SEEDS" "$WORK_ROOT/live_seeds.tmp" && mv "$WORK_ROOT/live_seeds.tmp" "$WORK_ROOT/live_seeds"
  fi
  if [[ ! -f "$WORK_ROOT/ground_truth.json" ]]; then
    (cd "$WORK_ROOT/harness_repo" && "$VENV_PY" -m src.rsa.outer.ground_truth --promoted "$PROMOTED" \
       --data "$EXISTING" --cache "$PROMOTE_WORK/.fit_cache" --out "$WORK_ROOT/ground_truth.json" \
       --rule "$GT_RULE" --promotion "$REPO/data/rsa/live_seeds/promotion.json" \
       --sweep "$REPO/data/rsa/sherlock_run2" --chain "$CHAIN_SEEDS")
  fi
) 9>"$WORK_ROOT/.setup.lock"
HARNESS_REPO="$WORK_ROOT/harness_repo"
SLURM_DIR="$HARNESS_REPO/scripts/rsa/slurm"
GT="$("$VENV_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["ground_truth"])' "$WORK_ROOT/ground_truth.json")"
GT_FILE="$("$VENV_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["ground_truth_file"])' "$WORK_ROOT/ground_truth.json")"
[[ -f "$GT_FILE" ]] || { echo "ERROR: no ground-truth file $GT_FILE" >&2; exit 1; }
# A starting model as ground truth is scrubbed from the tree's seeds; a run-2
# model (coherent) is in no seed set, and check_agent_tree.sh still refuses any
# copy of it or file named after it.
SCRUB=()
[[ -f "$HARNESS_REPO/$RSA_SEED_MODELS_REL/$GT.py" ]] && SCRUB=("$GT")

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
bash "$SLURM_DIR/build_agent_tree.sh" "$WORK_ROOT/agent_src" "$RUN_REPO" "$AGENT_PY" ${SCRUB[@]+"${SCRUB[@]}"}
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
echo "[rehearsal $SCOPE] chain $CHAIN, ${N_EXPERIMENTS} experiments, ${PARTICIPANTS:-200} people x 10 displays of ${DISPLAYS:-40}, $AGENT_MODEL -> $RUN_DIR"
cd "$HARNESS_REPO"
env ${UNSET_ARGS[@]+"${UNSET_ARGS[@]}"} PATH="$RUN_REPO/.agent_bin:$PATH" \
  "$AGENT_PY" -m src.rsa.outer.run --run-dir "$RUN_DIR" --private-dir "$CELL_DIR/private" \
    --seeds "$WORK_ROOT/live_seeds" --promoted "$PROMOTED" --existing-data "$EXISTING" --collection simulated --ground-truth "$GT_FILE" \
    --selection-scope "$SCOPE" --n-experiments "${N_EXPERIMENTS}" --participants "${PARTICIPANTS:-200}" \
    --trials 10 --displays "${DISPLAYS:-40}" --seed "$SEED" \
    --agent-root "$RUN_REPO" --agent-model "$AGENT_MODEL" --agent-timeout-sec "$AGENT_TIMEOUT_SEC"

# --- outputs: the run without fit caches, and the private records --------------------
rsync -a --delete --exclude '.fit_cache/' --exclude '.home/' --exclude '.agent_shell/' "$RUN_DIR"/ "$CELL_DIR/outputs"/
grep -rIlw -- "$GT" "$RUN_DIR"/experiment*/model_loop/round_* > "$CELL_DIR/gt_name_mentions.txt" 2>/dev/null || true
date -Is > "$CELL_DIR/DONE"
echo "[rehearsal $SCOPE] done: $CELL_DIR/outputs and $CELL_DIR/private"
