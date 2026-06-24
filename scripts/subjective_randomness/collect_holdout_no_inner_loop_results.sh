#!/bin/bash
# Copy the finished STANDARD (seed-model) holdout test-retest *no-inner-loop
# ablation* into the repo at data/results/holdout_no_inner_loop.
#
# This is the non-impossible companion to collect_no_inner_loop_results.sh: the
# ground truths are the held-out SEED models (bayesian_diagnosticity, ...), and
# the inner loop (candidate generation) is disabled, so each experiment fits only
# the seed set. Same artifact layout as every holdout test-retest run
# (test_retest.{json,csv,png} + run<r>/<gt>/holdout.{json,csv,png}), so it uses
# the same collector, collect_holdout_results.py.
#
# A thin, PINNED wrapper so this ablation's source/dest are explicit and
# reproducible. The collector copies only the lightweight artifacts, excludes the
# heavy .nc fit caches / per-task repo copies, distils a SUMMARY.md from
# test_retest.json, and fails loudly if expected artifacts are missing or the
# destination is already populated (pass --overwrite to replace, --summary-only
# for just the aggregate).
#
# NOTE on leakage: standard-holdout cells routinely carry leakage flags
# (any_mention / any_gt_named) because the held-out GT is a plausible theory the
# agents re-derive — the per-cell holdout.json records them, and the collected
# SUMMARY surfaces them. This is expected, not a collection error.
#
# Interpreter: the collector is pure-stdlib at its core plus tyro/pyprojroot for
# the CLI, all present in the repo's .venv — run that directly, no `uv` sync.
#
# Sherlock etiquette: we don't run Python on the login node. The copy is light
# I/O, but we still offload it to a short `dev` allocation via srun. Set LOCAL=1
# to run it inline (e.g. from an existing sh_dev shell or off-cluster).
#
# Usage:
#   bash scripts/subjective_randomness/collect_holdout_no_inner_loop_results.sh
#   bash scripts/subjective_randomness/collect_holdout_no_inner_loop_results.sh --overwrite
#   LOCAL=1 bash scripts/subjective_randomness/collect_holdout_no_inner_loop_results.sh
#
# Any flag after the script name is passed straight through to the collector.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

# --- this ablation's pinned parameters (override from the environment) -----
SOURCE="${SOURCE:-${SCRATCH:-$GROUP_SCRATCH}/auto-psych/holdout_test_retest_no_inner_loop}"
DEST="${DEST:-data/results/holdout_no_inner_loop}"
PY="${PY:-$REPO/.venv/bin/python}"
PARTITION="${PARTITION:-dev}"

[[ -d "$SOURCE" ]] || { echo "ERROR: source '$SOURCE' not found — set SOURCE=..." >&2; exit 1; }
[[ -f "$SOURCE/test_retest.json" ]] || { echo "ERROR: '$SOURCE' has no test_retest.json — did the analysis stage finish?" >&2; exit 1; }
[[ -x "$PY" ]] || { echo "ERROR: interpreter '$PY' not found — set PY=... (e.g. an sh_dev python)" >&2; exit 1; }

COLLECT=( "$PY" scripts/subjective_randomness/collect_holdout_results.py
          --source "$SOURCE"
          --dest   "$DEST"
          "$@" )

echo ">>> collect standard holdout test-retest (no-inner-loop ablation) results"
echo "    source: $SOURCE"
echo "    dest:   $REPO/$DEST"
echo "    python: $PY"
echo

if [[ -n "${LOCAL:-}" ]] || ! command -v srun >/dev/null 2>&1; then
  echo "[collect] running inline"
  exec "${COLLECT[@]}"
else
  echo "[collect] offloading to a short '$PARTITION' allocation (srun); set LOCAL=1 to run inline"
  exec srun --partition="$PARTITION" --time=00:20:00 --cpus-per-task=2 --mem=4GB \
    --job-name=collect_holdout_no_inner_loop "${COLLECT[@]}"
fi
