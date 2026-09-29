#!/bin/bash
# Submit the IMPOSSIBLE-theory holdout-recovery test-retest pipeline as chained
# jobs, as submit_holdout_test_retest.sh does for the literature sweep:
#   1. setup    - sync the uv env + stage the sweep's code once (harness_repo,
#                 agent_src, $WORK_ROOT/code_commit) and pristine, off-the-agent
#                 snapshots of the impossible recipe (models dir + config).
#                 Retries skip it and run on the staged code.
#   2. array    - R repeats x G impossible ground truths, each holding out ONE
#                 impossible model with a distinct per-repeat seed; the agent
#                 keeps the full normal seed pool and is expected to FAIL to
#                 recover the weird generator
#   3. retry    - resumes the array's failed tasks (holdout_retry.sbatch), up to
#                 MAX_RETRY_ROUNDS rounds, through this script
#   4. analysis - test-retest reliability summary and MISSING_CELLS.txt
#                 (reuses holdout_analysis.sbatch / holdout_test_retest.py —
#                 both are ground-truth-agnostic)
#
# Usage:
#   bash scripts/subjective_randomness/slurm/submit_impossible_holdout_test_retest.sh
#
# Override with env vars, e.g.:
#   N_REPEATS=5 BASE_SEED=100 MAX_PARALLEL=6 \
#   WORK_ROOT=$SCRATCH/auto-psych/impossible_run2 \
#     bash scripts/subjective_randomness/slurm/submit_impossible_holdout_test_retest.sh
set -euo pipefail

# Absolute dir of this script — passed to every job so they can find _env.sh.
HOLDOUT_SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HOLDOUT_SLURM_DIR
# The retry job resubmits failed tasks through the staged copy of this script.
export RETRY_SUBMIT_SCRIPT="submit_impossible_holdout_test_retest.sh"

# --- knobs (exported so --export=ALL carries them into every job) ----------
export N_REPEATS="${N_REPEATS:-5}"
# Must match the impossible config's gt_models keys (the setup job validates).
export GT_MODELS="${GT_MODELS:-more_heads_more_random fewer_heads_more_random longer_runs_more_random more_imbalance_more_random}"
export CONFIG="${CONFIG:-scripts/subjective_randomness/configs/impossible_holdout_recovery.yaml}"
read -r -a _GTS <<< "$GT_MODELS"
export N_GTS="${#_GTS[@]}"
TOTAL=$(( N_REPEATS * N_GTS ))
# Cap on simultaneous tasks (API rate + cores). Exported so a retry round's
# resubmission keeps the sweep's cap.
export MAX_PARALLEL="${MAX_PARALLEL:-5}"
ARRAY_TIME=""                       # use the sbatch directive's walltime
# The array's memory for one submission is ARRAY_MEM (below); MEM was this
# script's old name for it.
if [[ -n "${MEM:-}" ]]; then
  echo "ERROR: MEM is no longer read; set ARRAY_MEM (e.g. ARRAY_MEM=64G ARRAY_TASKS=16) instead." >&2
  exit 1
fi

# SMOKE=1: validate the whole chain cheaply — ONE task (repeat 1, first GT), one
# experiment, no inner-loop candidate rounds, tiny MCMC. Still exercises the repo
# copy, recipe/config exclusion, the opencode+Gemini agent, a PyMC fit, and the
# analysis.
if [[ -n "${SMOKE:-}" ]]; then
  TOTAL="${SMOKE_TASKS:-1}"; MAX_PARALLEL="$TOTAL"; ARRAY_TIME="02:00:00"
  export N_EXPERIMENTS="${N_EXPERIMENTS:-1}"
  export INNER_LOOP_ITERATIONS="${INNER_LOOP_ITERATIONS:-0}"
  export N_PARTICIPANTS="${N_PARTICIPANTS:-10}"
  export DRAWS="${DRAWS:-200}"; export TUNE="${TUNE:-200}"; export CHAINS="${CHAINS:-2}"
  export AGENT_TIMEOUT_SEC="${AGENT_TIMEOUT_SEC:-600}"
  echo ">>> SMOKE MODE: $TOTAL task(s), $MAX_PARALLEL concurrent, cheap settings"
fi

# Array spec: full sweep by default; ARRAY_TASKS overrides it to rerun a subset
# (e.g. ARRAY_TASKS=14 to redo one failed task on the same WORK_ROOT via --resume,
# or "1,4,14" / "1-5"). Pair with the original WORK_ROOT so --resume reuses work.
# The %MAX_PARALLEL cap applies to a subset too: a failure that hit every task
# (an API quota, a bad commit) must not resubmit all of them at once.
ARRAY_SPEC="1-${TOTAL}%${MAX_PARALLEL}"
[[ -n "${ARRAY_TASKS:-}" ]] && ARRAY_SPEC="${ARRAY_TASKS}%${MAX_PARALLEL}"
# ARRAY_MEM overrides the array's --mem (32GB) for this submission only (the
# retry job sets it for its out-of-memory group). It is passed as --mem, never
# through SBATCH_MEM_PER_NODE, which sbatch reads from the environment of every
# later job; unset here so no job inherits it (see submit_holdout_test_retest.sh).
array_mem="${ARRAY_MEM:-}"
unset ARRAY_MEM
# RETRY_ROUND > 0: a retry of this sweep's cells. It must run on the code the
# sweep staged, so it skips the setup job.
is_retry=""
(( ${RETRY_ROUND:-0} > 0 )) && is_retry=1

# Optional pass-throughs (only export if the caller set them).
[[ -n "${BASE_SEED:-}" ]] && export BASE_SEED
[[ -n "${REPO:-}"      ]] && export REPO
# Inner-loop ablation knob: INNER_LOOP_ITERATIONS=0 makes every array task pass
# --inner-loop-iterations 0, so the inner model loop only fits/scores the
# theorist's models (no candidate-conjecturing or critique agents are spawned).
# The array sbatch turns it into the CLI flag;
# run_impossible_no_inner_loop_test_retest.sh pins it to 0.
[[ -n "${INNER_LOOP_ITERATIONS:-}" ]] && export INNER_LOOP_ITERATIONS
# How claude agents are billed: CLAUDE_AUTH=subscription (CLAUDE_CODE_OAUTH_TOKEN)
# or CLAUDE_AUTH=api (ANTHROPIC_API_KEY), passed to --claude-auth. No default:
# with AGENT_BACKEND=claude a sweep that does not say is refused here, before
# anything is queued (and again by the harness, before any agent starts).
[[ -n "${CLAUDE_AUTH:-}" ]] && export CLAUDE_AUTH
if [[ "${AGENT_BACKEND:-}" == "claude" && -z "${CLAUDE_AUTH:-}" ]]; then
  echo "ERROR: AGENT_BACKEND=claude needs CLAUDE_AUTH=subscription or CLAUDE_AUTH=api" >&2
  exit 1
fi

# Keep Slurm logs off $HOME (15 GB, NFS). Mirror _env.sh's WORK_ROOT default,
# but in a dedicated impossible work root so it never collides with the standard
# holdout test-retest study.
export WORK_ROOT="${WORK_ROOT:-${SCRATCH:-$GROUP_SCRATCH}/auto-psych/impossible_holdout_test_retest}"
LOGDIR="$WORK_ROOT/slurm_logs"
mkdir -p "$LOGDIR"

cd "$HOLDOUT_SLURM_DIR"

setup_id=""
if [[ -n "$is_retry" ]]; then
  [[ -f "$WORK_ROOT/code_commit" ]] || { echo "ERROR: a retry needs the code the sweep staged, and $WORK_ROOT/code_commit does not exist" >&2; exit 1; }
  echo "retry round ${RETRY_ROUND}: no setup job; the cells resume on the staged code ($(cat "$WORK_ROOT/code_commit"))"
else
  setup_id=$(sbatch --parsable --export=ALL \
    --output="$LOGDIR/%x_%j.out" --error="$LOGDIR/%x_%j.out" \
    impossible_holdout_setup.sbatch)
  echo "submitted setup job:    $setup_id"
fi

array_id=$(sbatch --parsable ${setup_id:+--dependency=afterok:"$setup_id"} --export=ALL \
  --array="$ARRAY_SPEC" ${ARRAY_TIME:+--time="$ARRAY_TIME"} ${array_mem:+--mem="$array_mem"} \
  --output="$LOGDIR/%x_%A_%a.out" --error="$LOGDIR/%x_%A_%a.out" \
  impossible_holdout_recovery_array.sbatch)
echo "submitted array job:    $array_id ($ARRAY_SPEC${array_mem:+, --mem=$array_mem}; $N_REPEATS repeats x $N_GTS GTs)"

# Resume the array's failed tasks (timeouts, crashes, out-of-memory) once it
# finishes, up to MAX_RETRY_ROUNDS rounds (holdout_retry.sbatch, shared with
# the literature sweep). RETRY_ROUND counts the rounds already done; the retry
# job resubmits through this script (RETRY_SUBMIT_SCRIPT).
MAX_RETRY_ROUNDS="${MAX_RETRY_ROUNDS:-2}"
export MAX_RETRY_ROUNDS
if (( ${RETRY_ROUND:-0} < MAX_RETRY_ROUNDS )); then
  retry_id=$(sbatch --parsable --dependency=afterany:"$array_id" \
    --export=ALL,RETRY_ARRAY_ID="$array_id",RETRY_ROUND="${RETRY_ROUND:-0}" \
    --job-name=impossible_holdout_retry \
    --output="$LOGDIR/%x_%j.out" --error="$LOGDIR/%x_%j.out" \
    holdout_retry.sbatch)
  echo "submitted retry job:    $retry_id (round $(( ${RETRY_ROUND:-0} + 1 )) of $MAX_RETRY_ROUNDS)"
fi

# afterany: summarise once every task has finished, regardless of per-task
# success — the analysis uses whatever repeats produced a result and lists every
# expected cell without one in MISSING_CELLS.txt (each retry round submits its
# own, later summary). We reuse holdout_analysis.sbatch (ground-truth-agnostic)
# but give it the impossible job name so its logs are easy to spot.
analysis_id=$(sbatch --parsable --dependency=afterany:"$array_id" --export=ALL \
  --job-name=impossible_holdout_test_retest \
  --output="$LOGDIR/%x_%j.out" --error="$LOGDIR/%x_%j.out" \
  holdout_analysis.sbatch)
echo "submitted analysis job: $analysis_id"

echo
echo "watch with:  squeue --me"
echo "logs in:     $LOGDIR"
echo "results in:  $WORK_ROOT/test_retest.{json,csv,png}; missing cells in $WORK_ROOT/MISSING_CELLS.txt"
