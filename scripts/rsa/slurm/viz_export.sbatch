#!/bin/bash
#SBATCH --job-name=rsa_viz_export
#SBATCH --partition=mcfrank
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#
# Per-display choice counts and model predictions for the campaign figures
# (src.rsa.viz.export_panels; docs/auto_rsa/figures/README.md). Every model is fitted on the
# pooled data, which the consolidation's fit cache already holds (the finals, the bar, the lean
# model), so this mostly reads caches. Output: $WORK_ROOT/panels.json, aggregates only (no trial
# rows, no participant ids): bring it back as data/rsa/viz/panels.json.
set -euo pipefail
export REPO="${REPO:-${SLURM_SUBMIT_DIR:-$HOME/auto-psych}}"
export WORK_ROOT="${WORK_ROOT:-${SCRATCH:?}/auto-psych/rsa_consolidate}"
PREREG_ROOT="${PREREG_ROOT:-$SCRATCH/auto-psych/rsa_prereg}"
source "$REPO/scripts/rsa/slurm/_env.sh"
CODE_ROOT="$WORK_ROOT/code_viz"
mkdir -p "$CODE_ROOT"
WORK_ROOT="$CODE_ROOT" HARNESS_REPO="$CODE_ROOT/harness_repo" AGENT_SRC="$CODE_ROOT/agent_src" \
  bash "$REPO/scripts/rsa/slurm/stage_code.sh"
POOL="$WORK_ROOT/pool"
F="$REPO/data/rsa/live/finals"
P="$REPO/data/rsa/live_seeds/models"
S="$CODE_ROOT/harness_repo/src/pipelines/outer_loop/projects/rsa_reference/seed_models"
LEAN="$REPO/data/rsa/consolidate/eliminate/lean/cons_dup_sur_red_asp_int_comp_bind.py"
cd "$CODE_ROOT/harness_repo"
"$VENV_PY" -m src.rsa.viz.export_panels --pooled "$POOL/pooled.csv" \
  --prereg "$PREREG_ROOT/cells/prereg/run/experiment1/data/modelled.csv" --cache "$POOL/.fit_cache" \
  --models "c0=$F/boundary_avoidable_ambiguity_listener.py" "c1=$F/feature_surprisal_ensemble_listener.py" \
           "c2=$F/competition_endpoint_binding_lapse_l2.py" "lean=$LEAN" \
           "promoted=$P/rsa_l2_singleton_distinct_feat_color_l0.py" \
           "starting=$S/rsa_l1_shared_prior.py" "rsa_l1=$S/rsa_l1.py" \
  --out "$WORK_ROOT/panels.json"
echo "[viz export] done: $WORK_ROOT/panels.json (aggregates only) -> data/rsa/viz/panels.json"
