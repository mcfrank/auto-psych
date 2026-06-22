#!/bin/bash
# Launch ONE impossible-theory holdout-recovery test-retest run, with 5 repeats
# per held-out impossible ground-truth model.
#
# The impossible ground truths are deliberately weird generators of subjective
# randomness (e.g. "more heads => more random") that humans could not plausibly
# use. The agentic loop is seeded with the FULL normal seed set and is *expected
# to fail* to recover them — so the held-out correlation should stay low. Five
# repeats per ground truth let us see how *stably* it fails (test-retest of the
# null result), exactly mirroring run_test_retest.sh for the standard holdout.
#
# This is a thin, pinned wrapper around submit_impossible_holdout_test_retest.sh.
# It exists so the parameters for *this* run are explicit and reproducible, and
# so the run lands in its OWN work root (the array resumes with --resume, so
# reusing a stale root would silently skip already-present cells).
#
# What it does:
#   * 5 repeats x N impossible GTs = a (5*N)-task array (5 jobs per GT model),
#     each repeat r using --seed BASE_SEED+r so the synthetic data differs.
#   * GT_MODELS pinned to the impossible models; the setup job validates this
#     list against the config's gt_models and aborts (afterok) if they drift.
#   * full-strength config (configs/impossible_holdout_recovery.yaml:
#     n_experiments=3, inner loop 2x3, draws=2000/tune=1000/chains=4).
#   * the impossible recipe (models dir + config) is kept OFF the coding agent's
#     repo copy so it cannot read the answer; the parent loads the ground truth
#     from a pristine scratch snapshot instead (see the array sbatch header).
#
# Keep GT_MODELS below in sync with src/subjective_randomness/impossible_models/
# and the config's gt_models. The current set: more_heads_more_random,
# fewer_heads_more_random, longer_runs_more_random, more_imbalance_more_random.
#
# Usage:
#   # cheap pre-flight first (one task, no inner loop, tiny MCMC) — recommended
#   # after loop or impossible-model changes, to prove the whole chain still runs
#   # end to end and the recipe stays off the agent's disk:
#   SMOKE=1 bash scripts/subjective_randomness/slurm/run_impossible_test_retest.sh
#
#   # the real run:
#   bash scripts/subjective_randomness/slurm/run_impossible_test_retest.sh
#
# Any knob below can be overridden from the environment, e.g.:
#   N_REPEATS=3 BASE_SEED=200 bash scripts/.../run_impossible_test_retest.sh
set -euo pipefail

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- this run's pinned parameters ------------------------------------------
# 5 repeats per held-out model is the test-retest design.
export N_REPEATS="${N_REPEATS:-5}"
export BASE_SEED="${BASE_SEED:-100}"

# The impossible ground-truth models. Must match the config's gt_models keys
# exactly (the setup job enforces this and aborts the whole chain if they differ).
export GT_MODELS="${GT_MODELS:-more_heads_more_random fewer_heads_more_random longer_runs_more_random more_imbalance_more_random}"

# Full-strength impossible holdout config.
export CONFIG="${CONFIG:-scripts/subjective_randomness/configs/impossible_holdout_recovery.yaml}"

# A dedicated work root so we never resume a stale one. All run output, caches,
# the per-run venv, the pristine recipe snapshots, and the test-retest summary
# land here.
export WORK_ROOT="${WORK_ROOT:-${SCRATCH:-$GROUP_SCRATCH}/auto-psych/impossible_holdout_test_retest_full}"

# Cap on simultaneous array tasks — bounded by the Gemini API rate limit and by
# cores (each task asks for 8 CPUs / 32 GB). 5 keeps one full wave in flight.
export MAX_PARALLEL="${MAX_PARALLEL:-5}"

echo ">>> impossible holdout test-retest (all impossible ground-truth models)"
echo "    N_REPEATS=$N_REPEATS  BASE_SEED=$BASE_SEED  MAX_PARALLEL=$MAX_PARALLEL"
echo "    GT_MODELS=$GT_MODELS"
echo "    CONFIG=$CONFIG"
echo "    WORK_ROOT=$WORK_ROOT"
[[ -n "${SMOKE:-}" ]] && echo "    SMOKE MODE (cheap validation run)"
echo

exec bash "$SLURM_DIR/submit_impossible_holdout_test_retest.sh"
