#!/bin/bash
# Copy the finished IMPOSSIBLE-theory holdout test-retest *no-inner-loop ablation*
# into the repo at data/results/impossible_holdout_no_inner_loop.
#
# The ablation runs the same impossible-theory holdout test-retest but with the
# inner loop (candidate generation) disabled, so each experiment fits only the
# seed set — the companion to the full run in
# data/results/impossible_holdout_test_retest. It writes the identical artifact
# layout (test_retest.{json,csv,png} + run<r>/<gt>/holdout.{json,csv,png}).
#
# This is a thin, PINNED wrapper that just fixes this ablation's source/dest and
# delegates to collect_impossible_results.sh (the same collector the full run
# uses) — no copy logic is duplicated. That collector copies only the lightweight
# artifacts, excludes the heavy .nc fit caches / per-task repo copies, distils a
# SUMMARY.md, runs the repo .venv interpreter, offloads to a short `dev` srun
# (LOCAL=1 to run inline), and fails loudly if the dest is already populated.
#
# Usage:
#   bash scripts/subjective_randomness/collect_no_inner_loop_results.sh
#   bash scripts/subjective_randomness/collect_no_inner_loop_results.sh --overwrite
#   LOCAL=1 bash scripts/subjective_randomness/collect_no_inner_loop_results.sh
#
# Any flag after the script name is forwarded to the collector.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SOURCE="${SOURCE:-${SCRATCH:-$GROUP_SCRATCH}/auto-psych/impossible_holdout_test_retest_no_inner_loop}"
export DEST="${DEST:-data/results/impossible_holdout_no_inner_loop}"

exec bash "$DIR/collect_impossible_results.sh" "$@"
