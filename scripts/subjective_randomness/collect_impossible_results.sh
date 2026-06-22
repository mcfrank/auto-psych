#!/bin/bash
# Copy the finished IMPOSSIBLE-theory holdout test-retest run from scratch into
# the repo at data/results/impossible_holdout_test_retest.
#
# This is a thin, PINNED wrapper around scripts/subjective_randomness/
# collect_holdout_results.py — the same collector the standard holdout run uses,
# because the impossible test-retest produces the identical artifact layout
# (test_retest.{json,csv,png} + run<r>/<gt>/holdout.{json,csv,png}). It exists so
# the parameters for *this* run are explicit and reproducible, mirroring
# scripts/outer_loop_live/collect_results.sh for the human study.
#
# The collector copies only the lightweight result artifacts, EXCLUDES the heavy
# material we never commit (per-task repo copies, the shared venv, MCMC .nc fit
# caches), distils a SUMMARY.md from test_retest.json, and fails loudly if
# expected artifacts are missing or the destination is already populated (pass
# --overwrite to replace it, --summary-only to skip the per-run holdout.*).
#
# Interpreter: the collector is pure-stdlib at its core plus tyro/pyprojroot for
# the CLI, all present in the repo's .venv — so we run that interpreter directly
# and never need a `uv` env sync (slow/fragile on el7 compute nodes).
#
# Sherlock etiquette: we don't run Python on the login node. The copy is light
# I/O, but we still offload it to a short `dev` allocation via srun. Set LOCAL=1
# to run it inline instead (e.g. off-cluster, or from an existing sh_dev shell).
#
# Usage:
#   bash scripts/subjective_randomness/collect_impossible_results.sh
#   bash scripts/subjective_randomness/collect_impossible_results.sh --overwrite
#   bash scripts/subjective_randomness/collect_impossible_results.sh --summary-only
#   LOCAL=1 bash scripts/subjective_randomness/collect_impossible_results.sh
#
# Any flag after the script name is passed straight through to the collector.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

# --- this run's pinned parameters (override from the environment) ----------
# The _full work root written by run_impossible_test_retest.sh.
SOURCE="${SOURCE:-${SCRATCH:-$GROUP_SCRATCH}/auto-psych/impossible_holdout_test_retest_full}"
DEST="${DEST:-data/results/impossible_holdout_test_retest}"
# Repo .venv already has tyro + pyprojroot; the collector needs nothing heavier.
PY="${PY:-$REPO/.venv/bin/python}"
PARTITION="${PARTITION:-dev}"

[[ -d "$SOURCE" ]] || { echo "ERROR: source '$SOURCE' not found — set SOURCE=..." >&2; exit 1; }
[[ -f "$SOURCE/test_retest.json" ]] || { echo "ERROR: '$SOURCE' has no test_retest.json — did the analysis stage finish?" >&2; exit 1; }
[[ -x "$PY" ]] || { echo "ERROR: interpreter '$PY' not found — set PY=... (e.g. an sh_dev python)" >&2; exit 1; }

# Collector invocation; trailing "$@" forwards any pass-through flags.
COLLECT=( "$PY" scripts/subjective_randomness/collect_holdout_results.py
          --source "$SOURCE"
          --dest   "$DEST"
          "$@" )

echo ">>> collect impossible-theory holdout test-retest results"
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
    --job-name=collect_impossible_holdout "${COLLECT[@]}"
fi
