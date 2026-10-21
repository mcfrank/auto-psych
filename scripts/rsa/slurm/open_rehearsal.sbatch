#!/bin/bash
#SBATCH --job-name=rsa_open_rehearsal
#SBATCH --partition=mcfrank
#SBATCH --array=0
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=5-00:00:00
#SBATCH --output=%x_%A_%a.out
#
# The open campaign's Phase 3 rehearsal (docs/auto_open/PLAN.md §9): one chain of the open
# campaign on simulated participants. The designer (Gemini 4) and the theorists (Gemini 3.8 Flash)
# are real, sandboxed agents with no network; the pilot's participants are the harness's Gemini
# calls; the experiments' people answer from a hidden ground truth (src.rsa_open.ground_truth:
# the closed rehearsals' coherent truth, ported and fixed at its posterior means on E). Array task
# i runs chain CHAINS[i] (default "0": the truth was picked for chain 0). After promote.sbatch (it
# reads run 2's promote_work/all_trials.csv, E).
#
#   mkdir -p "$SCRATCH/auto-psych/rsa_open_rehearsal/logs"
#   sbatch --chdir="$HOME/auto-psych" -o "$SCRATCH/auto-psych/rsa_open_rehearsal/logs/%x_%A_%a.out" \
#     scripts/rsa/slurm/open_rehearsal.sbatch
#
# Each experiment: a design step of up to two designer cycles of 2 h of agent time each plus the
# harness's checks (EIG, link check, pilot: about 6 h in the worst case), simulated collection,
# then the inner loop (MAX_ITERATIONS rounds of CANDIDATES theorists). Resubmitting a task resumes
# its chain: finished stages are skipped, an interrupted design or loop restarts (its attempt
# kept as <stage>.abandoned_<i>).
#
# Nothing that names the ground truth enters the agents' tree: the truth and its record live in
# $WORK_ROOT/private (outside it), the tree is checked for its name before any agent starts, and
# environment variables that point outside the tree are withheld from the agents.
#
# Overrides: CHAINS ("0"), N_EXPERIMENTS (3), MAX_ITERATIONS (5), CANDIDATES (6),
# PILOT_PARTICIPANTS (12), THEORIST_TIMEOUT_SEC (2400), SEED (0), RUN2_ROOT (run 2's sweep root).
set -euo pipefail
export REPO="${REPO:-${SLURM_SUBMIT_DIR:-$HOME/auto-psych}}"
export WORK_ROOT="${WORK_ROOT:-${SCRATCH:?}/auto-psych/rsa_open_rehearsal}"
source "$REPO/scripts/rsa/slurm/_env.sh"
rsa_require_opencode
rsa_require_gemini_key
command -v bwrap >/dev/null || { echo "FATAL: no bwrap (sandboxed agents need it)" >&2; exit 1; }
command -v flock >/dev/null || { echo "FATAL: no flock" >&2; exit 1; }

TASK="${SLURM_ARRAY_TASK_ID:?run as an array task}"
read -r -a CHAIN_LIST <<< "${CHAINS:-0}"
CHAIN="${CHAIN_LIST[$TASK]:?array task $TASK has no chain in CHAINS=${CHAINS:-0}}"
N_EXPERIMENTS="${N_EXPERIMENTS:-3}"
MAX_ITERATIONS="${MAX_ITERATIONS:-5}"
CANDIDATES="${CANDIDATES:-6}"
PILOT_PARTICIPANTS="${PILOT_PARTICIPANTS:-12}"
THEORIST_TIMEOUT_SEC="${THEORIST_TIMEOUT_SEC:-2400}"
SEED="${SEED:-0}"
RUN2_ROOT="${RUN2_ROOT:-$(rsa_default_work_root)}"  # the run-2 sweep (_cells.sh)
EXISTING="$RUN2_ROOT/promote_work/all_trials.csv"   # E: every literature trial
SEEDS_SRC="$REPO/data/rsa_open/seeds"
CLOSED_RECORD="$REPO/data/rsa/rehearsal2/ground_truth.json"
PRIVATE="$WORK_ROOT/private"
for f in "$EXISTING" "$SEEDS_SRC/seeds.json" "$SEEDS_SRC/bar/models_manifest.yaml" "$CLOSED_RECORD"; do
  [[ -f "$f" ]] || { echo "ERROR: $f is missing (run promote.sbatch; pull auto-psych-open)" >&2; exit 1; }
done
mkdir -p "$WORK_ROOT" "$CELLS_ROOT" "$PRIVATE"

# --- once per rehearsal: staged code, the seeds, the ground truth ------------------
(
  flock 9
  bash "$REPO/scripts/rsa/slurm/stage_code.sh"
  if [[ ! -d "$WORK_ROOT/open_seeds" ]]; then
    cp -r "$SEEDS_SRC" "$WORK_ROOT/open_seeds.tmp" && mv "$WORK_ROOT/open_seeds.tmp" "$WORK_ROOT/open_seeds"
  fi
  if [[ ! -f "$PRIVATE/ground_truth/ground_truth.json" ]]; then
    (cd "$WORK_ROOT/harness_repo" && "$VENV_PY" -m src.rsa_open.ground_truth --bar "$WORK_ROOT/open_seeds/bar" \
       --out "$PRIVATE/ground_truth" --cache "$PRIVATE/.fit_cache" --literature "$EXISTING" --record "$CLOSED_RECORD")
  fi
) 9>"$WORK_ROOT/.setup.lock"
HARNESS_REPO="$WORK_ROOT/harness_repo"
SLURM_DIR="$HARNESS_REPO/scripts/rsa/slurm"
GT_DIR="$PRIVATE/ground_truth"
GT="$("$VENV_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["ground_truth"])' "$GT_DIR/ground_truth.json")"
GT_ORIGINAL="$("$VENV_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["original"])' "$GT_DIR/ground_truth.json")"
[[ -f "$GT_ORIGINAL" && -f "$GT_DIR/ground_truth.py" ]] || { echo "ERROR: the ground truth's files are missing" >&2; exit 1; }

# --- the cell and its agent tree (as outer_rehearsal.sbatch) ------------------------
CELL="open_chain_$CHAIN"
CELL_DIR="$CELLS_ROOT/$CELL"
mkdir -p "$CELL_DIR"
[[ -s "$CELL_DIR/agent_tree_id" ]] || head -c 8 /dev/urandom | od -An -tx1 | tr -d ' \n' > "$CELL_DIR/agent_tree_id"
AGENT_DIR="$AGENT_TREES_ROOT/$(cat "$CELL_DIR/agent_tree_id")"
RUN_REPO="$AGENT_DIR/repo"
RUN_DIR="$RUN_REPO/_runs/chain_$CHAIN"
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
bash "$SLURM_DIR/build_agent_tree.sh" "$WORK_ROOT/agent_src" "$RUN_REPO" "$AGENT_PY"
bash "$SLURM_DIR/check_agent_tree.sh" "$RUN_REPO" --gt-file "$GT_ORIGINAL"
# The ported truth is a different file with the same theory: no copy of it, and its name nowhere
# outside the runs (the chain's own records are checked after the run).
if grep -rIl --exclude-dir=_runs -- "$GT" "$RUN_REPO" >&2; then
  echo "ERROR: the agents' tree names the ground truth (files above)" >&2; exit 1
fi
if find "$RUN_REPO" -type f -name '*.py' -size "$(stat -c %s "$GT_DIR/ground_truth.py")c" \
     -exec cmp -s "$GT_DIR/ground_truth.py" {} \; -print | grep . >&2; then
  echo "ERROR: the agents' tree holds a copy of the ported ground truth (files above)" >&2; exit 1
fi

# --- the chain ----------------------------------------------------------------------
WITHHELD=()
for name in $(compgen -e); do
  value="${!name}"
  if [[ "$value" == *"$WORK_ROOT"* || "$value" == *"$REPO"* || "$value" == *"$RUN2_ROOT"* || "$value" == *"$GT"* ]]; then
    WITHHELD+=("$name")
  fi
done
UNSET_ARGS=()
for name in ${WITHHELD[@]+"${WITHHELD[@]}"}; do UNSET_ARGS+=(-u "$name"); done
echo "[open rehearsal] withheld from the agents' environment: ${WITHHELD[*]:-}"
echo "[open rehearsal] chain $CHAIN, $N_EXPERIMENTS experiments, $MAX_ITERATIONS rounds of $CANDIDATES theorists -> $RUN_DIR"
cd "$HARNESS_REPO"
env ${UNSET_ARGS[@]+"${UNSET_ARGS[@]}"} PATH="$RUN_REPO/.agent_bin:$PATH" \
  "$AGENT_PY" -m src.rsa_open.run_chain --chain "$CHAIN" --run-dir "$RUN_DIR" --private-dir "$GT_DIR" \
    --agent-root "$RUN_REPO" --literature "$EXISTING" --seeds "$WORK_ROOT/open_seeds" \
    --n-experiments "$N_EXPERIMENTS" --max-iterations "$MAX_ITERATIONS" --candidate-count "$CANDIDATES" \
    --pilot-participants "$PILOT_PARTICIPANTS" --theorist-timeout-sec "$THEORIST_TIMEOUT_SEC" --seed "$SEED" \
    --code-commit "$(cat "$WORK_ROOT/code_commit")"

# --- outputs: the run without fit caches; where agents' files name the truth ----------
rsync -a --delete --exclude '.fit_cache/' --exclude '.home/' --exclude '.agent_shell/' "$RUN_DIR"/ "$CELL_DIR/outputs"/
grep -rIlw -- "$GT" "$RUN_DIR"/experiment*/loop/round_* "$RUN_DIR"/experiment*/design/cycle_*/work \
  > "$CELL_DIR/gt_name_mentions.txt" 2>/dev/null || true
date -Is > "$CELL_DIR/DONE"
echo "[open rehearsal] done: $CELL_DIR/outputs (the truth: $GT_DIR)"
