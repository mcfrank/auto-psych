#!/bin/bash
# ABLATION: holdout-recovery test-retest with the INNER LOOP REMOVED.
#
# This is the standard holdout test-retest design (run_test_retest.sh) with one
# change: every run sets --inner-loop-iterations 0, so the inner model loop runs
# ZERO candidate-conjecturing rounds. At 0 iterations the inner loop spawns no
# agents at all — it only fits and ELPD-LOO-scores the theorist's own models
# (the experiment-1 seed set, and whatever the theory agent proposes in later
# experiments), then records the single seed-scoring step the trajectory needs.
# Everything else is identical to the full pipeline, so the only thing ablated
# is the inner candidate/critique search:
#
#   full pipeline:  theory -> design -> collect -> [inner loop: fit + critique +
#                   conjecture N rounds of new PyMC models] -> score
#   this ablation:  theory -> design -> collect -> [fit + score the theorist's
#                   models only]                              -> score
#
# The outer-loop agents (theory in experiments >= 2, design every experiment)
# still run in full — only the inner model-discovery loop is gone. Comparing
# this against run_test_retest.sh isolates how much of the recovery comes from
# the inner loop's agent-discovered models vs. the outer loop alone.
#
# This is a thin, pinned wrapper around submit_holdout_test_retest.sh. It lands
# in its OWN work root (suffix `_no_inner_loop`) so it never resumes or collides
# with the full-pipeline study.
#
# Usage:
#   # cheap pre-flight first (one task, tiny MCMC) — recommended after loop or
#   # seed-model changes, to prove the chain still runs end to end:
#   SMOKE=1 bash scripts/subjective_randomness/slurm/run_no_inner_loop_test_retest.sh
#
#   # the real ablation run:
#   bash scripts/subjective_randomness/slurm/run_no_inner_loop_test_retest.sh
#
# Any knob below can be overridden from the environment, e.g.:
#   N_REPEATS=3 BASE_SEED=200 bash scripts/.../run_no_inner_loop_test_retest.sh
set -euo pipefail

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- the ablation: zero inner-loop iterations ------------------------------
# Forwarded by submit_holdout_test_retest.sh -> the array sbatch, which turns it
# into `--inner-loop-iterations 0` on every holdout_recovery.py invocation.
export INNER_LOOP_ITERATIONS="${INNER_LOOP_ITERATIONS:-0}"

# --- this run's pinned parameters (match run_test_retest.sh) ----------------
# 5 repeats per held-out model is the test-retest design.
export N_REPEATS="${N_REPEATS:-5}"
export BASE_SEED="${BASE_SEED:-100}"

# The current seed models. Must match the config's gt_models keys exactly (the
# setup job enforces this and aborts the whole chain if they differ).
export GT_MODELS="${GT_MODELS:-bayesian_diagnosticity encoding_compressibility prototype_similarity window_typicality}"

# Full-strength holdout config (only inner_loop.max_iterations is overridden, to
# 0, via INNER_LOOP_ITERATIONS above).
export CONFIG="${CONFIG:-scripts/subjective_randomness/configs/holdout_recovery.yaml}"

# A dedicated work root so the ablation never resumes/collides with the full run.
export WORK_ROOT="${WORK_ROOT:-${SCRATCH:-$GROUP_SCRATCH}/auto-psych/holdout_test_retest_no_inner_loop}"

# Cap on simultaneous array tasks (Gemini API rate + cores). 5 keeps one full
# wave in flight.
export MAX_PARALLEL="${MAX_PARALLEL:-5}"

echo ">>> holdout test-retest ABLATION — inner loop removed (--inner-loop-iterations 0)"
echo "    INNER_LOOP_ITERATIONS=$INNER_LOOP_ITERATIONS"
echo "    N_REPEATS=$N_REPEATS  BASE_SEED=$BASE_SEED  MAX_PARALLEL=$MAX_PARALLEL"
echo "    GT_MODELS=$GT_MODELS"
echo "    CONFIG=$CONFIG"
echo "    WORK_ROOT=$WORK_ROOT"
[[ -n "${SMOKE:-}" ]] && echo "    SMOKE MODE (cheap validation run)"
echo

exec bash "$SLURM_DIR/submit_holdout_test_retest.sh"
