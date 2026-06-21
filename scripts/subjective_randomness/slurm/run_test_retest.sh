#!/bin/bash
# Launch ONE holdout-recovery test-retest run over the current seed-model set,
# with 5 repeats per held-out model.
#
# This is a thin, pinned wrapper around submit_holdout_test_retest.sh. It exists
# so the parameters for *this* run are explicit and reproducible, and so the run
# lands in its OWN work root instead of the default one (which still holds an
# older smoke run; because the array resumes with --resume, reusing that root
# would silently skip already-present cells and reuse stale results).
#
# What it does:
#   * 5 repeats x N held-out GTs = a (5*N)-task array (5 jobs per seed model),
#     each repeat r using --seed BASE_SEED+r so the synthetic data differs.
#   * GT_MODELS pinned to the current seed models; the setup job validates this
#     list against the config's gt_models and aborts (afterok) if they drift.
#   * full-strength config (scripts/.../configs/holdout_recovery.yaml:
#     n_experiments=3, inner loop 2x3, draws=2000/tune=1000/chains=4).
#
# Keep GT_MODELS below in sync with the seed_models dir / config gt_models. The
# current set: bayesian_diagnosticity, encoding_compressibility,
# prototype_similarity, window_typicality.
#
# Usage:
#   # cheap pre-flight first (one task, no inner loop, tiny MCMC) — recommended
#   # after loop or seed-model changes, to prove the whole chain still runs end
#   # to end:
#   SMOKE=1 bash scripts/subjective_randomness/slurm/run_test_retest.sh
#
#   # the real run:
#   bash scripts/subjective_randomness/slurm/run_test_retest.sh
#
# Any knob below can be overridden from the environment, e.g.:
#   N_REPEATS=3 BASE_SEED=200 bash scripts/.../run_test_retest.sh
set -euo pipefail

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- this run's pinned parameters ------------------------------------------
# 5 repeats per held-out model is the test-retest design.
export N_REPEATS="${N_REPEATS:-5}"
export BASE_SEED="${BASE_SEED:-100}"

# The current seed models. Must match the config's gt_models keys exactly (the
# setup job enforces this and aborts the whole chain if they differ).
export GT_MODELS="${GT_MODELS:-bayesian_diagnosticity encoding_compressibility prototype_similarity window_typicality}"

# Full-strength holdout config.
export CONFIG="${CONFIG:-scripts/subjective_randomness/configs/holdout_recovery.yaml}"

# A dedicated work root so we never resume the stale default one. All run
# output, caches, the per-run venv, and the test-retest summary land here.
export WORK_ROOT="${WORK_ROOT:-${SCRATCH:-$GROUP_SCRATCH}/auto-psych/holdout_test_retest_full}"

# Cap on simultaneous array tasks — bounded by the Gemini API rate limit and by
# cores (each task asks for 8 CPUs / 32 GB). 5 keeps one full wave in flight.
export MAX_PARALLEL="${MAX_PARALLEL:-5}"

echo ">>> holdout test-retest (all current seed models)"
echo "    N_REPEATS=$N_REPEATS  BASE_SEED=$BASE_SEED  MAX_PARALLEL=$MAX_PARALLEL"
echo "    GT_MODELS=$GT_MODELS"
echo "    CONFIG=$CONFIG"
echo "    WORK_ROOT=$WORK_ROOT"
[[ -n "${SMOKE:-}" ]] && echo "    SMOKE MODE (cheap validation run)"
echo

exec bash "$SLURM_DIR/submit_holdout_test_retest.sh"
