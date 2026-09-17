#!/bin/bash
#SBATCH --job-name=evaluate_recovery
#SBATCH --partition=normal
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --output=%x_%j.out
#
# Evaluate a finished recovery sweep: verify, compare against prior sweeps,
# run oracle diagnostics, and generate the recovery report.
#
# Required env vars (passed via --export):
#   WORK_ROOT   - sweep root (contains run1..runN)
#   REPO        - repo checkout (for scripts and src)
#   VENV_PY     - path to the venv Python interpreter
#   ITER2_SWEEP - iteration 2 sweep root (for comparison)
#   ITER3_SWEEP - iteration 3 sweep root (for comparison)
#   BASELINE_SWEEP - pre-campaign baseline sweep root (for comparison)
#
# Each step fails loudly: set -e ensures a failure stops the job.
set -euo pipefail

: "${WORK_ROOT:?WORK_ROOT must be set}"
: "${REPO:?REPO must be set}"
: "${VENV_PY:?VENV_PY must be set}"
: "${ITER2_SWEEP:?ITER2_SWEEP must be set}"
: "${ITER3_SWEEP:?ITER3_SWEEP must be set}"
: "${BASELINE_SWEEP:?BASELINE_SWEEP must be set}"

ANALYSIS_DIR="${WORK_ROOT}/analysis"
mkdir -p "$ANALYSIS_DIR"

cd "$REPO"

echo "=== Evaluation started at $(date) ==="
echo "WORK_ROOT=$WORK_ROOT"
echo "REPO=$REPO"
echo "VENV_PY=$VENV_PY"
echo "ITER2_SWEEP=$ITER2_SWEEP"
echo "ITER3_SWEEP=$ITER3_SWEEP"
echo "BASELINE_SWEEP=$BASELINE_SWEEP"
echo ""

# --- Step a: verifier --------------------------------------------------------
echo "=== Step a: Running verifier on sweep ==="
export ALL_JOB_IDS="${ALL_JOB_IDS:-0}"
WORK_ROOT="$WORK_ROOT" bash scripts/subjective_randomness/slurm/verify_holdout_run.sh
echo "Verifier completed."
echo ""

# --- Step b: compare_matched_cells -------------------------------------------
echo "=== Step b: Comparing against prior sweeps ==="

echo "  Comparing vs iter2..."
$VENV_PY scripts/subjective_randomness/compare_matched_cells.py \
    --sweep-a "$ITER2_SWEEP" \
    --sweep-b "$WORK_ROOT" \
    --out "$ANALYSIS_DIR/vs_iter2"
echo ""

echo "  Comparing vs iter3..."
$VENV_PY scripts/subjective_randomness/compare_matched_cells.py \
    --sweep-a "$ITER3_SWEEP" \
    --sweep-b "$WORK_ROOT" \
    --out "$ANALYSIS_DIR/vs_iter3"
echo ""

echo "  Comparing vs baseline..."
$VENV_PY scripts/subjective_randomness/compare_matched_cells.py \
    --sweep-a "$BASELINE_SWEEP" \
    --sweep-b "$WORK_ROOT" \
    --out "$ANALYSIS_DIR/vs_baseline"
echo ""

# --- Step c: oracle diagnostics ----------------------------------------------
echo "=== Step c: Running oracle diagnostics ==="
for holdout_json in "$WORK_ROOT"/run*/*/holdout.json; do
    cell_dir="$(dirname "$holdout_json")"
    cell_name="$(basename "$(dirname "$cell_dir")")/$(basename "$cell_dir")"
    echo "  Oracle: $cell_name"
    $VENV_PY scripts/subjective_randomness/oracle_admitted_models.py \
        --result "$holdout_json" \
        --steps final
done
echo ""

# --- Step d: recovery report -------------------------------------------------
echo "=== Step d: Generating recovery report ==="
$VENV_PY scripts/subjective_randomness/recovery_report.py \
    --sweep "$WORK_ROOT" \
    --label consolidated_raw \
    --compare "vs_iter2=$ANALYSIS_DIR/vs_iter2" \
    --compare "vs_iter3=$ANALYSIS_DIR/vs_iter3" \
    --compare "vs_baseline=$ANALYSIS_DIR/vs_baseline" \
    --oracle-glob "$WORK_ROOT/run*/*/oracle.json" \
    --out "$ANALYSIS_DIR/RESULTS_auto.md"

echo ""
echo "=== Evaluation completed at $(date) ==="
echo "Results: $ANALYSIS_DIR/RESULTS_auto.md"
