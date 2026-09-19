#!/bin/bash
#SBATCH --job-name=reanalyze_final
#SBATCH --partition=normal
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --output=%x_%j.out
#
# Offline re-analysis over both recovery sweeps using the fixed analysis tools
# (P29: archive extraction, common-pool construction).
#
# Required env vars (passed via --export):
#   SWEEP1_ROOT    - first sweep root (20 cells, candidate-write bug)
#   SWEEP2_ROOT    - second sweep root (15 cells, fix applied)
#   REPO           - repo checkout (for scripts and src)
#   VENV_PY        - path to the venv Python interpreter
#   ITER2_SWEEP    - iteration 2 sweep root (confounded reference)
#   ITER3_SWEEP    - iteration 3 sweep root (confounded reference)
#   BASELINE_SWEEP - pre-campaign baseline sweep root (confounded reference)
#   ANALYSIS_DIR   - output directory for all analysis products
#
set -euo pipefail

: "${SWEEP1_ROOT:?SWEEP1_ROOT must be set}"
: "${SWEEP2_ROOT:?SWEEP2_ROOT must be set}"
: "${REPO:?REPO must be set}"
: "${VENV_PY:?VENV_PY must be set}"
: "${ITER2_SWEEP:?ITER2_SWEEP must be set}"
: "${ITER3_SWEEP:?ITER3_SWEEP must be set}"
: "${BASELINE_SWEEP:?BASELINE_SWEEP must be set}"
: "${ANALYSIS_DIR:?ANALYSIS_DIR must be set}"

mkdir -p "$ANALYSIS_DIR"

cd "$REPO"

echo "=== Re-analysis started at $(date) ==="
echo "SWEEP1_ROOT=$SWEEP1_ROOT"
echo "SWEEP2_ROOT=$SWEEP2_ROOT"
echo "REPO=$REPO"
echo "ANALYSIS_DIR=$ANALYSIS_DIR"
echo "ITER2_SWEEP=$ITER2_SWEEP"
echo "ITER3_SWEEP=$ITER3_SWEEP"
echo "BASELINE_SWEEP=$BASELINE_SWEEP"
echo ""

# Track which steps failed so the job continues and reports everything it can.
FAILURES=""

# --- Step 1: Isolating comparison (sweep2 vs sweep1) -------------------------
echo "=== Step 1: Isolating comparison (sweep_rerun vs sweep) ==="
if $VENV_PY scripts/subjective_randomness/compare_matched_cells.py \
    --sweep-a "$SWEEP1_ROOT" \
    --sweep-b "$SWEEP2_ROOT" \
    --out "$ANALYSIS_DIR/rerun_vs_sweep1"; then
    echo "  OK"
else
    echo "  FAILED (exit $?)"
    FAILURES="$FAILURES isolating_comparison"
fi
echo ""

# --- Step 2: Confounded references for sweep_rerun ---------------------------
echo "=== Step 2: Confounded references for sweep_rerun ==="

echo "  sweep_rerun vs iter2..."
if $VENV_PY scripts/subjective_randomness/compare_matched_cells.py \
    --sweep-a "$ITER2_SWEEP" \
    --sweep-b "$SWEEP2_ROOT" \
    --out "$ANALYSIS_DIR/sweep2_vs_iter2"; then
    echo "  OK"
else
    echo "  FAILED (exit $?) — older archives may not be reconstructable"
    FAILURES="$FAILURES sweep2_vs_iter2"
fi
echo ""

echo "  sweep_rerun vs iter3..."
if $VENV_PY scripts/subjective_randomness/compare_matched_cells.py \
    --sweep-a "$ITER3_SWEEP" \
    --sweep-b "$SWEEP2_ROOT" \
    --out "$ANALYSIS_DIR/sweep2_vs_iter3"; then
    echo "  OK"
else
    echo "  FAILED (exit $?) — older archives may not be reconstructable"
    FAILURES="$FAILURES sweep2_vs_iter3"
fi
echo ""

echo "  sweep_rerun vs baseline..."
if $VENV_PY scripts/subjective_randomness/compare_matched_cells.py \
    --sweep-a "$BASELINE_SWEEP" \
    --sweep-b "$SWEEP2_ROOT" \
    --out "$ANALYSIS_DIR/sweep2_vs_baseline"; then
    echo "  OK"
else
    echo "  FAILED (exit $?) — older archives may not be reconstructable"
    FAILURES="$FAILURES sweep2_vs_baseline"
fi
echo ""

# --- Step 2b: Confounded references for sweep1 -------------------------------
echo "=== Step 2b: Confounded references for sweep1 ==="

echo "  sweep vs iter2..."
if $VENV_PY scripts/subjective_randomness/compare_matched_cells.py \
    --sweep-a "$ITER2_SWEEP" \
    --sweep-b "$SWEEP1_ROOT" \
    --out "$ANALYSIS_DIR/sweep1_vs_iter2"; then
    echo "  OK"
else
    echo "  FAILED (exit $?) — older archives may not be reconstructable"
    FAILURES="$FAILURES sweep1_vs_iter2"
fi
echo ""

echo "  sweep vs iter3..."
if $VENV_PY scripts/subjective_randomness/compare_matched_cells.py \
    --sweep-a "$ITER3_SWEEP" \
    --sweep-b "$SWEEP1_ROOT" \
    --out "$ANALYSIS_DIR/sweep1_vs_iter3"; then
    echo "  OK"
else
    echo "  FAILED (exit $?) — older archives may not be reconstructable"
    FAILURES="$FAILURES sweep1_vs_iter3"
fi
echo ""

echo "  sweep vs baseline..."
if $VENV_PY scripts/subjective_randomness/compare_matched_cells.py \
    --sweep-a "$BASELINE_SWEEP" \
    --sweep-b "$SWEEP1_ROOT" \
    --out "$ANALYSIS_DIR/sweep1_vs_baseline"; then
    echo "  OK"
else
    echo "  FAILED (exit $?) — older archives may not be reconstructable"
    FAILURES="$FAILURES sweep1_vs_baseline"
fi
echo ""

# --- Step 3: Oracle diagnostics on every cell of both sweeps -----------------
echo "=== Step 3: Oracle diagnostics ==="

echo "  Oracle on sweep1..."
for holdout_json in "$SWEEP1_ROOT"/run*/*/holdout.json; do
    cell_dir="$(dirname "$holdout_json")"
    cell_name="$(basename "$(dirname "$cell_dir")")/$(basename "$cell_dir")"
    echo "    $cell_name"
    if ! $VENV_PY scripts/subjective_randomness/oracle_admitted_models.py \
        --result "$holdout_json" \
        --steps final; then
        echo "    FAILED (exit $?)"
        FAILURES="$FAILURES oracle_sweep1_${cell_name}"
    fi
done
echo ""

echo "  Oracle on sweep_rerun..."
for holdout_json in "$SWEEP2_ROOT"/run*/*/holdout.json; do
    cell_dir="$(dirname "$holdout_json")"
    cell_name="$(basename "$(dirname "$cell_dir")")/$(basename "$cell_dir")"
    echo "    $cell_name"
    if ! $VENV_PY scripts/subjective_randomness/oracle_admitted_models.py \
        --result "$holdout_json" \
        --steps final; then
        echo "    FAILED (exit $?)"
        FAILURES="$FAILURES oracle_sweep2_${cell_name}"
    fi
done
echo ""

# --- Step 4: Recovery reports ------------------------------------------------
echo "=== Step 4: Recovery reports ==="

# Build comparison args for sweep1 report
COMPARE_ARGS_S1=()
[[ -d "$ANALYSIS_DIR/rerun_vs_sweep1" ]] && COMPARE_ARGS_S1+=(--compare "vs_sweep_rerun=$ANALYSIS_DIR/rerun_vs_sweep1")
[[ -d "$ANALYSIS_DIR/sweep1_vs_iter2" ]] && COMPARE_ARGS_S1+=(--compare "vs_iter2=$ANALYSIS_DIR/sweep1_vs_iter2")
[[ -d "$ANALYSIS_DIR/sweep1_vs_iter3" ]] && COMPARE_ARGS_S1+=(--compare "vs_iter3=$ANALYSIS_DIR/sweep1_vs_iter3")
[[ -d "$ANALYSIS_DIR/sweep1_vs_baseline" ]] && COMPARE_ARGS_S1+=(--compare "vs_baseline=$ANALYSIS_DIR/sweep1_vs_baseline")

echo "  Report for sweep1..."
$VENV_PY scripts/subjective_randomness/recovery_report.py \
    --sweep "$SWEEP1_ROOT" \
    --label "sweep1_raw" \
    "${COMPARE_ARGS_S1[@]}" \
    --oracle-glob "$SWEEP1_ROOT/run*/*/oracle.json" \
    --out "$ANALYSIS_DIR/RESULTS_sweep1.md"
echo ""

# Build comparison args for sweep2 (rerun) report
COMPARE_ARGS_S2=()
[[ -d "$ANALYSIS_DIR/rerun_vs_sweep1" ]] && COMPARE_ARGS_S2+=(--compare "vs_sweep1=$ANALYSIS_DIR/rerun_vs_sweep1")
[[ -d "$ANALYSIS_DIR/sweep2_vs_iter2" ]] && COMPARE_ARGS_S2+=(--compare "vs_iter2=$ANALYSIS_DIR/sweep2_vs_iter2")
[[ -d "$ANALYSIS_DIR/sweep2_vs_iter3" ]] && COMPARE_ARGS_S2+=(--compare "vs_iter3=$ANALYSIS_DIR/sweep2_vs_iter3")
[[ -d "$ANALYSIS_DIR/sweep2_vs_baseline" ]] && COMPARE_ARGS_S2+=(--compare "vs_baseline=$ANALYSIS_DIR/sweep2_vs_baseline")

echo "  Report for sweep_rerun..."
$VENV_PY scripts/subjective_randomness/recovery_report.py \
    --sweep "$SWEEP2_ROOT" \
    --label "sweep2_rerun_raw" \
    "${COMPARE_ARGS_S2[@]}" \
    --oracle-glob "$SWEEP2_ROOT/run*/*/oracle.json" \
    --out "$ANALYSIS_DIR/RESULTS_sweep2.md"
echo ""

# --- Summary -----------------------------------------------------------------
echo "=== Re-analysis completed at $(date) ==="
echo "Analysis dir: $ANALYSIS_DIR"
if [[ -n "$FAILURES" ]]; then
    echo "FAILURES:$FAILURES"
    echo "NOTE: some steps failed — see output above for details."
else
    echo "All steps completed successfully."
fi
