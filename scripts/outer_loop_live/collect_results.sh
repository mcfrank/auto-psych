#!/bin/bash
# Copy the finished live (human) outer-loop study from scratch into the repo at
# data/results/human_experiment.
#
# This is a thin, PINNED wrapper around the collector that does the real work,
# scripts/outer_loop_live/collect_human_results.py. It exists so the parameters
# for *this* study are explicit and reproducible (mirroring run_test_retest.sh's
# relationship to submit_holdout_test_retest.sh). The collector:
#   * copies only the lightweight result artifacts (responses, design, cognitive
#     models, model-loop results, agent logs),
#   * EXCLUDES the heavy material we never commit (per-run repo copies, language
#     caches, the multi-100MB MCMC .nc fit caches),
#   * SCRUBS every Prolific id (the participant_id_str column and bare 24-hex
#     worker/study ids in logs/manifests/transcripts) — and fails loudly if any
#     id would survive into the repo, and
#   * writes a generated SUMMARY.md (winning model, posterior, ELPD margin per
#     (run, experiment)).
# The collector also fails loudly if the destination is already populated; pass
# --overwrite (below) to replace it.
#
# Interpreter: the collector is pure-stdlib at its core plus tyro/pyprojroot for
# the CLI, all present in the repo's .venv — so we run that interpreter directly
# and never need a `uv` env sync (which is slow/fragile on el7 compute nodes).
#
# Sherlock etiquette: we don't run Python on the login node. The copy is light
# I/O, but we still offload it to a short `dev` allocation via srun. Set LOCAL=1
# to run it inline instead (e.g. off-cluster, or from an existing sh_dev shell).
#
# Usage:
#   bash scripts/outer_loop_live/collect_results.sh                 # collect run1-3
#   bash scripts/outer_loop_live/collect_results.sh --overwrite     # replace dest
#   LOCAL=1 bash scripts/outer_loop_live/collect_results.sh         # run inline
#   RUNS="run3" bash scripts/outer_loop_live/collect_results.sh     # just one run
#
# Any flag after the script name is passed straight through to the collector
# (e.g. --overwrite, --include-pilots).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

# --- this study's pinned parameters (override from the environment) --------
SOURCE="${SOURCE:-${SCRATCH:-$GROUP_SCRATCH}/auto-psych/outer_loop_live}"
DEST="${DEST:-data/results/human_experiment}"
# The three real runs of this study (pilots / _validate* are skipped anyway).
RUNS="${RUNS:-run1 run2 run3}"
# Repo .venv already has tyro + pyprojroot; the collector needs nothing heavier.
PY="${PY:-$REPO/.venv/bin/python}"
PARTITION="${PARTITION:-dev}"

[[ -d "$SOURCE" ]] || { echo "ERROR: source '$SOURCE' not found — set SOURCE=..." >&2; exit 1; }
[[ -x "$PY" ]]     || { echo "ERROR: interpreter '$PY' not found — set PY=... (e.g. an sh_dev python)" >&2; exit 1; }

# Collector invocation. `--runs $RUNS` is intentionally unquoted so the space-
# separated labels split into separate args; trailing "$@" forwards any
# pass-through flags (e.g. --overwrite).
COLLECT=( "$PY" scripts/outer_loop_live/collect_human_results.py
          --source "$SOURCE"
          --dest   "$DEST"
          --runs   $RUNS
          "$@" )

echo ">>> collect live human-loop results"
echo "    source: $SOURCE"
echo "    dest:   $REPO/$DEST"
echo "    runs:   $RUNS"
echo "    python: $PY"
echo

if [[ -n "${LOCAL:-}" ]] || ! command -v srun >/dev/null 2>&1; then
  echo "[collect] running inline"
  exec "${COLLECT[@]}"
else
  echo "[collect] offloading to a short '$PARTITION' allocation (srun); set LOCAL=1 to run inline"
  exec srun --partition="$PARTITION" --time=00:20:00 --cpus-per-task=2 --mem=4GB \
    --job-name=collect_human_results "${COLLECT[@]}"
fi
