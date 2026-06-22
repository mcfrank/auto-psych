#!/bin/bash
# Submit the IMPOSSIBLE-theory holdout-recovery test-retest pipeline as three
# chained jobs:
#   1. setup    - sync the uv env + stage pristine, off-the-agent snapshots of
#                 the impossible recipe (models dir + config) on scratch
#   2. array    - R repeats x G impossible ground truths, each holding out ONE
#                 impossible model with a distinct per-repeat seed; the agent
#                 keeps the full normal seed pool and is expected to FAIL to
#                 recover the weird generator
#   3. analysis - test-retest reliability summary (reuses holdout_analysis.sbatch
#                 / holdout_test_retest.py — both are ground-truth-agnostic)
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

# --- knobs (exported so --export=ALL carries them into every job) ----------
export N_REPEATS="${N_REPEATS:-5}"
# Must match the impossible config's gt_models keys (the setup job validates).
export GT_MODELS="${GT_MODELS:-more_heads_more_random fewer_heads_more_random longer_runs_more_random more_imbalance_more_random}"
export CONFIG="${CONFIG:-scripts/subjective_randomness/configs/impossible_holdout_recovery.yaml}"
read -r -a _GTS <<< "$GT_MODELS"
export N_GTS="${#_GTS[@]}"
TOTAL=$(( N_REPEATS * N_GTS ))
MAX_PARALLEL="${MAX_PARALLEL:-5}"   # cap simultaneous tasks (API rate + cores)
ARRAY_TIME=""                       # use the sbatch directive's walltime

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
ARRAY_SPEC="1-${TOTAL}%${MAX_PARALLEL}"
[[ -n "${ARRAY_TASKS:-}" ]] && ARRAY_SPEC="$ARRAY_TASKS"

# Optional pass-throughs (only export if the caller set them).
[[ -n "${BASE_SEED:-}" ]] && export BASE_SEED
[[ -n "${REPO:-}"      ]] && export REPO

# Keep Slurm logs off $HOME (15 GB, NFS). Mirror _env.sh's WORK_ROOT default,
# but in a dedicated impossible work root so it never collides with the standard
# holdout test-retest study.
export WORK_ROOT="${WORK_ROOT:-${SCRATCH:-$GROUP_SCRATCH}/auto-psych/impossible_holdout_test_retest}"
LOGDIR="$WORK_ROOT/slurm_logs"
mkdir -p "$LOGDIR"

cd "$HOLDOUT_SLURM_DIR"

setup_id=$(sbatch --parsable --export=ALL \
  --output="$LOGDIR/%x_%j.out" --error="$LOGDIR/%x_%j.out" \
  impossible_holdout_setup.sbatch)
echo "submitted setup job:    $setup_id"

array_id=$(sbatch --parsable --dependency=afterok:"$setup_id" --export=ALL \
  --array="$ARRAY_SPEC" ${ARRAY_TIME:+--time="$ARRAY_TIME"} \
  --output="$LOGDIR/%x_%A_%a.out" --error="$LOGDIR/%x_%A_%a.out" \
  impossible_holdout_recovery_array.sbatch)
echo "submitted array job:    $array_id (1-$TOTAL%$MAX_PARALLEL  =  $N_REPEATS repeats x $N_GTS GTs)"

# afterany: summarise once every task has finished, regardless of per-task
# success — the analysis uses whatever repeats produced a tidy CSV (agent runs
# are flaky over many hours). Use afterok to instead require all tasks to pass.
# We reuse holdout_analysis.sbatch (ground-truth-agnostic) but give it the
# impossible job name so its logs are easy to spot.
analysis_id=$(sbatch --parsable --dependency=afterany:"$array_id" --export=ALL \
  --job-name=impossible_holdout_test_retest \
  --output="$LOGDIR/%x_%j.out" --error="$LOGDIR/%x_%j.out" \
  holdout_analysis.sbatch)
echo "submitted analysis job: $analysis_id"

echo
echo "watch with:  squeue --me"
echo "logs in:     $LOGDIR"
echo "results in:  $WORK_ROOT/test_retest.{json,csv,png}"
