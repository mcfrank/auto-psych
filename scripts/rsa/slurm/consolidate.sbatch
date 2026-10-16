#!/bin/bash
#SBATCH --job-name=rsa_consolidate
#SBATCH --partition=mcfrank
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --time=7-00:00:00
#SBATCH --output=%x_%j.out
#
# Consolidating the three finals into the leanest justifiable model
# (docs/auto_rsa/CONSOLIDATION.md). No people, no money, no agents.
#
#   STAGE=eliminate  backward elimination over the superset by grouped CV on the campaign's pooled
#                    data -> $WORK_ROOT/eliminate/ (elimination.json, lean/). Resubmitting resumes:
#                    every model's CV is saved.
#   STAGE=score      the lean model, the full superset and the three finals, fitted on the pooled
#                    data, scored on the pre-registered experiment's modelled trials ->
#                    $WORK_ROOT/score.json (sums only)
#
# The pooled data and their fit cache are copied once from the pre-registration's work root, so the
# finals' fits are reused; the pre-registered experiment's data are read only by STAGE=score.
set -euo pipefail
export REPO="${REPO:-${SLURM_SUBMIT_DIR:-$HOME/auto-psych}}"
export WORK_ROOT="${WORK_ROOT:-${SCRATCH:?}/auto-psych/rsa_consolidate}"
PREREG_ROOT="${PREREG_ROOT:-$SCRATCH/auto-psych/rsa_prereg}"
source "$REPO/scripts/rsa/slurm/_env.sh"
STAGE="${STAGE:?set STAGE=eliminate or score}"
[[ "$STAGE" =~ ^(eliminate|score)$ ]] || { echo "ERROR: STAGE must be eliminate or score, not $STAGE" >&2; exit 1; }
mkdir -p "$WORK_ROOT"
CODE_ROOT="$WORK_ROOT/code_$STAGE"
mkdir -p "$CODE_ROOT"
WORK_ROOT="$CODE_ROOT" bash "$REPO/scripts/rsa/slurm/stage_code.sh"
POOL="$WORK_ROOT/pool"
if [[ ! -f "$POOL/pooled.csv" ]]; then
  mkdir -p "$POOL.tmp"
  cp "$PREREG_ROOT/pool/pooled.csv" "$POOL.tmp/pooled.csv"
  rsync -a "$PREREG_ROOT/pool/.fit_cache/" "$POOL.tmp/.fit_cache/"
  mv "$POOL.tmp" "$POOL"
fi
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
cd "$CODE_ROOT/harness_repo"
case "$STAGE" in
  eliminate)
    "$VENV_PY" -m src.rsa.consolidate.eliminate --data "$POOL/pooled.csv" --out "$WORK_ROOT/eliminate" \
      --cache "$POOL/.fit_cache" --workers "$WORKERS"
    ;;
  score)
    TEST="$PREREG_ROOT/cells/prereg/run/experiment1/data/modelled.csv"
    LEAN="$(ls "$WORK_ROOT"/eliminate/lean/cons_*.py)"
    [[ $(echo "$LEAN" | wc -l) -eq 1 ]] || { echo "ERROR: expected one lean model in $WORK_ROOT/eliminate/lean" >&2; exit 1; }
    FULL="$(ls "$WORK_ROOT"/eliminate/models/cons_d2_dup_rich_emp_sur_red_asp_int_avo_comp_conf_bind_att.py)"
    F="$REPO/data/rsa/live/finals"
    "$VENV_PY" -m src.rsa.consolidate.score --train "$POOL/pooled.csv" --test "$TEST" --cache "$POOL/.fit_cache" \
      --models "lean=$LEAN" "full=$FULL" "c0=$F/boundary_avoidable_ambiguity_listener.py" \
               "c1=$F/feature_surprisal_ensemble_listener.py" "c2=$F/competition_endpoint_binding_lapse_l2.py" \
               "promoted=$REPO/data/rsa/live_seeds/models/rsa_l2_singleton_distinct_feat_color_l0.py" \
      --reference c0 --out "$WORK_ROOT/score.json" --workers 6
    ;;
esac
echo "[consolidate $STAGE] done"
