#!/bin/bash
# Submit the holdout-recovery test-retest pipeline as three chained jobs:
#   1. setup    - sync the uv env + stage a pristine GT-model snapshot on scratch
#   2. array    - R repeats x G ground truths tasks, each holding out ONE model
#                 with a distinct per-repeat seed
#   3. analysis - test-retest reliability summary, runs after the array
#
# Usage:
#   bash scripts/subjective_randomness/slurm/submit_holdout_test_retest.sh
#
# Override with env vars, e.g.:
#   N_REPEATS=5 BASE_SEED=100 MAX_PARALLEL=6 \
#   WORK_ROOT=$SCRATCH/auto-psych/tr_run2 \
#     bash scripts/subjective_randomness/slurm/submit_holdout_test_retest.sh
set -euo pipefail

# Absolute dir of this script — passed to every job so they can find _env.sh.
HOLDOUT_SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HOLDOUT_SLURM_DIR

# --- knobs (exported so --export=ALL carries them into every job) ----------
export N_REPEATS="${N_REPEATS:-5}"
# Must match the config's gt_models keys (the setup job validates this).
export GT_MODELS="${GT_MODELS:-bayesian_diagnosticity encoding_compressibility prototype_similarity window_typicality}"
export SEED_MODELS_REL="${SEED_MODELS_REL:-src/pipelines/outer_loop/projects/subjective_randomness/seed_models}"
read -r -a _GTS <<< "$GT_MODELS"
export N_GTS="${#_GTS[@]}"
TOTAL=$(( N_REPEATS * N_GTS ))
MAX_PARALLEL="${MAX_PARALLEL:-5}"   # cap simultaneous tasks (API rate + cores)
ARRAY_TIME=""                       # use the sbatch directive's walltime

# SMOKE=1: validate the whole chain cheaply — ONE task (repeat 1, first GT),
# one experiment, no inner-loop candidate rounds, tiny MCMC. Still exercises the
# repo copy, GT removal, the opencode+Gemini agent, a PyMC fit, and the analysis.
if [[ -n "${SMOKE:-}" ]]; then
  # SMOKE_TASKS tasks (default 1), all run concurrently so a multi-task smoke
  # also exercises cross-task isolation (opencode DBs, repo copies).
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
[[ -n "${CONFIG:-}"    ]] && export CONFIG
[[ -n "${REPO:-}"      ]] && export REPO

# Keep Slurm logs off $HOME (15 GB, NFS). Mirror _env.sh's WORK_ROOT default.
export WORK_ROOT="${WORK_ROOT:-${SCRATCH:-$GROUP_SCRATCH}/auto-psych/holdout_test_retest}"
LOGDIR="$WORK_ROOT/slurm_logs"
mkdir -p "$LOGDIR"

cd "$HOLDOUT_SLURM_DIR"

setup_id=$(sbatch --parsable --export=ALL \
  --output="$LOGDIR/%x_%j.out" --error="$LOGDIR/%x_%j.out" \
  holdout_setup.sbatch)
echo "submitted setup job:    $setup_id"

array_id=$(sbatch --parsable --dependency=afterok:"$setup_id" --export=ALL \
  --array="$ARRAY_SPEC" ${ARRAY_TIME:+--time="$ARRAY_TIME"} \
  --output="$LOGDIR/%x_%A_%a.out" --error="$LOGDIR/%x_%A_%a.out" \
  holdout_recovery_array.sbatch)
echo "submitted array job:    $array_id (1-$TOTAL%$MAX_PARALLEL  =  $N_REPEATS repeats x $N_GTS GTs)"

# afterany: summarise once every task has finished, regardless of per-task
# success — the analysis uses whatever repeats produced a tidy CSV (agent runs
# are flaky over many hours). Use afterok to instead require all tasks to pass.
analysis_id=$(sbatch --parsable --dependency=afterany:"$array_id" --export=ALL \
  --output="$LOGDIR/%x_%j.out" --error="$LOGDIR/%x_%j.out" \
  holdout_analysis.sbatch)
echo "submitted analysis job: $analysis_id"

echo
echo "watch with:  squeue --me"
echo "logs in:     $LOGDIR"
echo "results in:  $WORK_ROOT/test_retest.{json,csv,png}"
