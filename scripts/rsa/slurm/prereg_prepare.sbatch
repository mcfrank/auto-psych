#!/bin/bash
#SBATCH --job-name=rsa_prereg_prepare
#SBATCH --partition=mcfrank
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#
# The pre-registered final experiment's preparation (docs/auto_rsa/PREREG_FINAL.md,
# docs/auto_rsa/HANDOFF_prereg.md). No people, no money, no agents.
#
#   STAGE=fit_bar  the 17 bar models (5 starting + 12 promoted) fitted once on the
#                  finals' pooled data, as the finals were: -> $WORK_ROOT/bar_pool_preds.json
#                  (bring it back as data/rsa/live/bar_pool_preds.json)
#   STAGE=power    the power grid on random designs within the quotas -> $WORK_ROOT/power.json
#   STAGE=design   the design at DISPLAYS and PARTICIPANTS (half EIG, half random), then its
#                  own power -> $WORK_ROOT/design/ and $WORK_ROOT/design_power.json
#
# Each stage runs on its own staged copy of the code ($WORK_ROOT/code_<stage>, with its
# code_commit), since every step's results are pushed before the next stage: the
# live job's own staging ($WORK_ROOT/harness_repo) is the frozen code alone. A stage
# re-run on newer code needs RESTAGE_REASON (stage_code.sh).
#
# The pooled data and the finals' fits are copied once from the campaign
# (CAMPAIGN_ROOT/finals_pool: pooled.csv and .fit_cache), so every fit is on the
# same data file and the finals' fits are reused, not repeated.
set -euo pipefail
export REPO="${REPO:-${SLURM_SUBMIT_DIR:-$HOME/auto-psych}}"
export WORK_ROOT="${WORK_ROOT:-${SCRATCH:?}/auto-psych/rsa_prereg}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-$SCRATCH/auto-psych/rsa_campaign}"
source "$REPO/scripts/rsa/slurm/_env.sh"
STAGE="${STAGE:?set STAGE=fit_bar, power or design}"
[[ "$STAGE" =~ ^(fit_bar|power|design)$ ]] || { echo "ERROR: STAGE must be fit_bar, power or design, not $STAGE" >&2; exit 1; }
mkdir -p "$WORK_ROOT"
CODE_ROOT="$WORK_ROOT/code_$STAGE"
mkdir -p "$CODE_ROOT"
# _env.sh exported HARNESS_REPO and AGENT_SRC under the prereg root, and stage_code.sh
# keeps them when set: name the stage's own (job 47895766 staged into $WORK_ROOT/harness_repo).
WORK_ROOT="$CODE_ROOT" HARNESS_REPO="$CODE_ROOT/harness_repo" AGENT_SRC="$CODE_ROOT/agent_src" \
  bash "$REPO/scripts/rsa/slurm/stage_code.sh"
POOL="$WORK_ROOT/pool"
if [[ ! -f "$POOL/pooled.csv" ]]; then
  mkdir -p "$POOL.tmp"
  cp "$CAMPAIGN_ROOT/finals_pool/pooled.csv" "$POOL.tmp/pooled.csv"
  rsync -a "$CAMPAIGN_ROOT/finals_pool/.fit_cache/" "$POOL.tmp/.fit_cache/"
  mv "$POOL.tmp" "$POOL"
fi
FINALS_PREDS="$REPO/data/rsa/live/finals_pool_preds_final.json"
BAR_PREDS="${BAR_PREDS:-$REPO/data/rsa/live/bar_pool_preds.json}"
STARTING="$CODE_ROOT/harness_repo/src/pipelines/outer_loop/projects/rsa_reference/seed_models"
PROMOTED="$REPO/data/rsa/live_seeds/models"
cd "$CODE_ROOT/harness_repo"
case "$STAGE" in
  fit_bar)
    "$VENV_PY" -m src.rsa.prereg.pool_fit --data "$POOL/pooled.csv" --cache "$POOL/.fit_cache" \
      --models "seed:$STARTING" "promoted:$PROMOTED" --workers "${SLURM_CPUS_PER_TASK:-8}" \
      --out "$WORK_ROOT/bar_pool_preds.json"
    ;;
  power)
    [[ -f "$BAR_PREDS" ]] || { echo "ERROR: $BAR_PREDS is missing (STAGE=fit_bar, then bring it back)" >&2; exit 1; }
    "$VENV_PY" -m src.rsa.prereg.power --predictions "$FINALS_PREDS" "$BAR_PREDS" \
      --displays ${DISPLAYS_GRID:-104 208 312 416} --participants ${PARTICIPANTS_GRID:-800 1200 1600 2400} \
      --n-sims "${N_SIMS:-2000}" --out "$WORK_ROOT/power.json"
    ;;
  design)
    : "${DISPLAYS:?set DISPLAYS (from the power simulation)}" "${PARTICIPANTS:?set PARTICIPANTS}"
    [[ -f "$BAR_PREDS" ]] || { echo "ERROR: $BAR_PREDS is missing" >&2; exit 1; }
    [[ ! -e "$WORK_ROOT/design/prereg_design.json" ]] || { echo "ERROR: $WORK_ROOT/design exists: a design is made once" >&2; exit 1; }
    "$VENV_PY" -m src.rsa.prereg.design --finals "$REPO/data/rsa/live/finals" --data "$POOL/pooled.csv" \
      --cache "$POOL/.fit_cache" --starting "$STARTING" --promoted "$PROMOTED" \
      --displays "$DISPLAYS" --participants "$PARTICIPANTS" --out "$WORK_ROOT/design"
    "$VENV_PY" -m src.rsa.prereg.power --predictions "$FINALS_PREDS" "$BAR_PREDS" \
      --design "$WORK_ROOT/design/design.json" --participants "$PARTICIPANTS" \
      --n-sims "${N_SIMS:-2000}" --out "$WORK_ROOT/design_power.json"
    ;;
  *) echo "ERROR: STAGE must be fit_bar, power or design, not $STAGE" >&2; exit 1 ;;
esac
echo "[prereg $STAGE] done"
