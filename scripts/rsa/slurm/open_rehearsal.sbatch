#!/bin/bash
#SBATCH --job-name=rsa_open_rehearsal
#SBATCH --partition=mcfrank
#SBATCH --array=0
#SBATCH --cpus-per-task=24
#SBATCH --mem=96GB
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
# ARM ("open"; "fixed": the closed forced-choice paradigm every experiment, no designer, cell
# fixed_chain_<k>; both arms can share a WORK_ROOT and its ground truth). FIXED_SPEC=pair (with
# ARM=fixed): the two-measure rehearsal variant of the fixed paradigm (src/rsa_open/fixed.py
# build_pair), cell fixed_pair_chain_<k>.
#
# Overrides: CHAINS ("0"), N_EXPERIMENTS (4), MAX_ITERATIONS (5), CANDIDATES (6),
# PILOT_PARTICIPANTS (12), CV_FOLDS (5), THEORIST_TIMEOUT_SEC (3600), SEED (0), RUN2_ROOT (run 2's
# sweep root), LITERATURE ("all": E, run 2's all_trials.csv; "pragmods": the committed pragmods
# trials alone, about 7x fewer, for a dry run that tests the machinery, not the science),
# REPLAY_FROM (another rehearsal's WORK_ROOT) and REPLAY_EXPERIMENTS ("1"): before the chain, copy
# those experiments of that rehearsal's chain, everything but their loops, into this run, so the
# chain skips their design and collection and reruns only their loops on this code (a replay of
# the same data, PI 2026-10-24). Copy that rehearsal's private/ground_truth into this WORK_ROOT
# first: the replayed data were simulated from it.
set -euo pipefail
export REPO="${REPO:-${SLURM_SUBMIT_DIR:-$HOME/auto-psych}}"
export WORK_ROOT="${WORK_ROOT:-${SCRATCH:?}/auto-psych/rsa_open_rehearsal}"
source "$REPO/scripts/rsa/slurm/_env.sh"
rsa_require_opencode
rsa_require_gemini_key
command -v bwrap >/dev/null || { echo "FATAL: no bwrap (sandboxed agents need it)" >&2; exit 1; }
command -v flock >/dev/null || { echo "FATAL: no flock" >&2; exit 1; }
# Before any staging, fit or agent: the venv has every package and module a chain uses (the first
# submission failed on google-genai, pydantic and resvg-py, missing from the shared venv).
(cd "$REPO" && "$VENV_PY" -m src.rsa_open.env_check) || {
  echo "FATAL: the venv cannot run the open campaign; on a login node: source scripts/rsa/slurm/_env.sh &&" \
       "source scripts/_venv.sh && build_shared_venv" >&2; exit 1; }

TASK="${SLURM_ARRAY_TASK_ID:?run as an array task}"
read -r -a CHAIN_LIST <<< "${CHAINS:-0}"
CHAIN="${CHAIN_LIST[$TASK]:?array task $TASK has no chain in CHAINS=${CHAINS:-0}}"
N_EXPERIMENTS="${N_EXPERIMENTS:-4}"
MAX_ITERATIONS="${MAX_ITERATIONS:-5}"
CANDIDATES="${CANDIDATES:-6}"
PILOT_PARTICIPANTS="${PILOT_PARTICIPANTS:-12}"
THEORIST_TIMEOUT_SEC="${THEORIST_TIMEOUT_SEC:-3600}"
SEED="${SEED:-0}"
RUN2_ROOT="${RUN2_ROOT:-$(rsa_default_work_root)}"  # the run-2 sweep (_cells.sh)
CV_FOLDS="${CV_FOLDS:-5}"
case "${LITERATURE:-all}" in
  all) EXISTING="$RUN2_ROOT/promote_work/all_trials.csv" ;;   # E: every literature trial
  pragmods) EXISTING="$REPO/src/pipelines/outer_loop/projects/rsa_reference/data/pragmods_trials.csv" ;;
  *) echo "ERROR: LITERATURE must be all or pragmods, not ${LITERATURE}" >&2; exit 1 ;;
esac
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
ARM="${ARM:-open}"
case "$ARM" in open|fixed) ;; *) echo "ERROR: ARM must be open or fixed, not $ARM" >&2; exit 1 ;; esac
SPEC_FLAG=()
SPEC_LABEL=""
case "${FIXED_SPEC:-}" in
  "") ;;
  pair) [[ "$ARM" == fixed ]] || { echo "ERROR: FIXED_SPEC is for ARM=fixed" >&2; exit 1; }
        SPEC_FLAG=(--fixed-spec "$WORK_ROOT/harness_repo/src/rsa_open/fixed_pair_paradigm.json"); SPEC_LABEL=_pair ;;
  *) echo "ERROR: FIXED_SPEC must be pair or unset, not $FIXED_SPEC" >&2; exit 1 ;;
esac
CELL="${ARM}${SPEC_LABEL}_chain_$CHAIN"
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

# --- a replay: earlier experiments' designs and data, without their loops -----------------
if [[ -n "${REPLAY_FROM:-}" ]]; then
  SRC_TREE_ID="$REPLAY_FROM/cells/$CELL/agent_tree_id"
  [[ -s "$SRC_TREE_ID" ]] || { echo "ERROR: $SRC_TREE_ID is missing (REPLAY_FROM=$REPLAY_FROM)" >&2; exit 1; }
  SRC_RUN="$AGENT_TREES_ROOT/$(cat "$SRC_TREE_ID")/repo/_runs/chain_$CHAIN"
  cmp -s "$REPLAY_FROM/private/ground_truth/ground_truth.json" "$GT_DIR/ground_truth.json" || {
    echo "ERROR: this WORK_ROOT's ground truth is not REPLAY_FROM's (copy its private/ground_truth first)" >&2; exit 1; }
  for n in ${REPLAY_EXPERIMENTS:-1}; do
    [[ -f "$SRC_RUN/experiment$n/data/rows.jsonl" ]] || { echo "ERROR: $SRC_RUN/experiment$n has no data" >&2; exit 1; }
    if [[ ! -d "$RUN_DIR/experiment$n" ]]; then
      mkdir -p "$RUN_DIR"
      rsync -a --exclude '/loop/' --exclude '/loop.abandoned_*/' "$SRC_RUN/experiment$n"/ "$RUN_DIR/experiment$n.tmp"/
      mv "$RUN_DIR/experiment$n.tmp" "$RUN_DIR/experiment$n"
      echo "[open rehearsal] replay: experiment $n's design and data from $SRC_RUN"
    fi
  done
  unset REPLAY_FROM SRC_TREE_ID SRC_RUN  # not for the agents' environment
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
echo "[open rehearsal] $ARM arm, chain $CHAIN, $N_EXPERIMENTS experiments, $MAX_ITERATIONS rounds of $CANDIDATES theorists," \
     "$CV_FOLDS folds, literature ${LITERATURE:-all} ($EXISTING) -> $RUN_DIR"
cd "$HARNESS_REPO"
env ${UNSET_ARGS[@]+"${UNSET_ARGS[@]}"} PATH="$RUN_REPO/.agent_bin:$PATH" \
  "$AGENT_PY" -m src.rsa_open.run_chain --chain "$CHAIN" --arm "$ARM" --run-dir "$RUN_DIR" --private-dir "$GT_DIR" \
    --agent-root "$RUN_REPO" --literature "$EXISTING" --seeds "$WORK_ROOT/open_seeds" \
    --n-experiments "$N_EXPERIMENTS" --max-iterations "$MAX_ITERATIONS" --candidate-count "$CANDIDATES" \
    --pilot-participants "$PILOT_PARTICIPANTS" --cv-folds "$CV_FOLDS" --theorist-timeout-sec "$THEORIST_TIMEOUT_SEC" --seed "$SEED" \
    ${SPEC_FLAG[@]+"${SPEC_FLAG[@]}"} \
    --code-commit "$(cat "$WORK_ROOT/code_commit")"

# --- outputs: the run without fit caches; where agents' files name the truth ----------
rsync -a --delete --exclude '.fit_cache/' --exclude '.home/' --exclude '.agent_shell/' "$RUN_DIR"/ "$CELL_DIR/outputs"/
grep -rIlw -- "$GT" "$RUN_DIR"/experiment*/loop/round_* "$RUN_DIR"/experiment*/design/cycle_*/work \
  > "$CELL_DIR/gt_name_mentions.txt" 2>/dev/null || true
date -Is > "$CELL_DIR/DONE"
echo "[open rehearsal] done: $CELL_DIR/outputs (the truth: $GT_DIR)"
