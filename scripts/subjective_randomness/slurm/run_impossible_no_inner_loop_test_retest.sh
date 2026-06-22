#!/bin/bash
# ABLATION: impossible-theory holdout-recovery test-retest with the INNER LOOP
# REMOVED.
#
# This is the impossible holdout test-retest design (run_impossible_test_retest.sh)
# with one change: every run sets --inner-loop-iterations 0, so the inner model
# loop runs ZERO candidate-conjecturing rounds. At 0 iterations the inner loop
# spawns no agents at all — it only fits and ELPD-LOO-scores the theorist's own
# models, then records the single seed-scoring step the trajectory needs.
#
# The impossible ground truths are deliberately weird generators of subjective
# randomness that humans could not plausibly use; the loop keeps the FULL normal
# seed set and is *expected to fail* to recover them. Removing the inner loop
# tests how that failure looks WITHOUT the agent's candidate model search: the
# outer-loop theory/design agents still run, but the inner conjecture/critique
# rounds that might otherwise stumble onto the weird generator are gone. Compare
# against run_impossible_test_retest.sh to isolate the inner loop's contribution.
#
# Leak prevention is identical to the full impossible run: the impossible recipe
# (models dir + config names) stays off every agent's repo copy, and the parent
# loads the ground truth from a pristine scratch snapshot.
#
# This is a thin, pinned wrapper around submit_impossible_holdout_test_retest.sh.
# It lands in its OWN work root (suffix `_no_inner_loop`) so it never resumes or
# collides with the full-pipeline impossible study.
#
# Usage:
#   # cheap pre-flight first (one task, tiny MCMC) — recommended after loop or
#   # impossible-model changes, to prove the chain still runs end to end and the
#   # recipe stays off the agent's disk:
#   SMOKE=1 bash scripts/subjective_randomness/slurm/run_impossible_no_inner_loop_test_retest.sh
#
#   # the real ablation run:
#   bash scripts/subjective_randomness/slurm/run_impossible_no_inner_loop_test_retest.sh
#
# Any knob below can be overridden from the environment, e.g.:
#   N_REPEATS=3 BASE_SEED=200 bash scripts/.../run_impossible_no_inner_loop_test_retest.sh
set -euo pipefail

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- the ablation: zero inner-loop iterations ------------------------------
# Forwarded by submit_impossible_holdout_test_retest.sh -> the array sbatch,
# which turns it into `--inner-loop-iterations 0` on every
# impossible_holdout_recovery.py invocation.
export INNER_LOOP_ITERATIONS="${INNER_LOOP_ITERATIONS:-0}"

# --- this run's pinned parameters (match run_impossible_test_retest.sh) ------
# 5 repeats per held-out model is the test-retest design.
export N_REPEATS="${N_REPEATS:-5}"
export BASE_SEED="${BASE_SEED:-100}"

# The impossible ground-truth models. Must match the config's gt_models keys
# exactly (the setup job enforces this and aborts the whole chain if they differ).
export GT_MODELS="${GT_MODELS:-more_heads_more_random fewer_heads_more_random longer_runs_more_random more_imbalance_more_random}"

# Full-strength impossible holdout config (only inner_loop.max_iterations is
# overridden, to 0, via INNER_LOOP_ITERATIONS above).
export CONFIG="${CONFIG:-scripts/subjective_randomness/configs/impossible_holdout_recovery.yaml}"

# A dedicated work root so the ablation never resumes/collides with the full run.
export WORK_ROOT="${WORK_ROOT:-${SCRATCH:-$GROUP_SCRATCH}/auto-psych/impossible_holdout_test_retest_no_inner_loop}"

# Cap on simultaneous array tasks (Gemini API rate + cores). 5 keeps one full
# wave in flight.
export MAX_PARALLEL="${MAX_PARALLEL:-5}"

echo ">>> impossible holdout test-retest ABLATION — inner loop removed (--inner-loop-iterations 0)"
echo "    INNER_LOOP_ITERATIONS=$INNER_LOOP_ITERATIONS"
echo "    N_REPEATS=$N_REPEATS  BASE_SEED=$BASE_SEED  MAX_PARALLEL=$MAX_PARALLEL"
echo "    GT_MODELS=$GT_MODELS"
echo "    CONFIG=$CONFIG"
echo "    WORK_ROOT=$WORK_ROOT"
[[ -n "${SMOKE:-}" ]] && echo "    SMOKE MODE (cheap validation run)"
echo

exec bash "$SLURM_DIR/submit_impossible_holdout_test_retest.sh"
