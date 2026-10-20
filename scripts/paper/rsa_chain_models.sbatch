#!/bin/bash
#SBATCH --partition=normal
#SBATCH --output=%x_%A_%a.out
# ============================================================================
#  Reference games: every other model of each live chain scored on the
#  pre-registered experiment (scripts/paper/rsa_chain_models.py), for the
#  paper's Figure 2. Descriptive only: the pre-registered data are spent.
#  Submit all three stages with scripts/paper/submit_rsa_chain_models.sh:
#
#    STAGE=setup  build the shared venv (scripts/_venv.sh), stage
#                 the code once ($WORK_ROOT/code, code_commit) and copy the
#                 pre-registration's pooled data
#    STAGE=fit    one shard of the ~220 chain models (an array task): each
#                 fitted on the pooled data with the bar's settings, its
#                 frozen pool predictions saved as a record
#    STAGE=score  every record scored on the pre-registered experiment's
#                 modelled trials, after checking the scoring reproduces the
#                 finals' and the bar's pre-registered lpd; sums only, to
#                 $REPO/data/rsa/prereg/chain_models_lpd.json
#
#  The models' files are read from the checkout's data/rsa/ (the staged code
#  has no data/). A resubmitted fit reuses every record and fit.
# ============================================================================
set -euo pipefail
export REPO="${REPO:-${SLURM_SUBMIT_DIR:-$HOME/auto-psych}}"
export WORK_ROOT="${WORK_ROOT:-${SCRATCH:?}/auto-psych/rsa_chain_models}"
PREREG_ROOT="${PREREG_ROOT:-$SCRATCH/auto-psych/rsa_prereg}"
source "$REPO/scripts/rsa/slurm/_env.sh"
STAGE="${STAGE:?set STAGE=setup, fit or score}"
CODE_ROOT="$WORK_ROOT/code"
POOL="$WORK_ROOT/pool"
D="$REPO/data/rsa"
COMMON=(--live-root "$D/live" --chain-seeds "$D/live_seeds/chains/chains.json"
        --prereg-analysis "$D/prereg/run/prereg_analysis.json" --work-dir "$WORK_ROOT/work")
case "$STAGE" in
  setup)
    build_shared_venv
    mkdir -p "$CODE_ROOT"
    # _env.sh exported HARNESS_REPO and AGENT_SRC under WORK_ROOT; name this job's own.
    WORK_ROOT="$CODE_ROOT" HARNESS_REPO="$CODE_ROOT/harness_repo" AGENT_SRC="$CODE_ROOT/agent_src" \
      bash "$REPO/scripts/rsa/slurm/stage_code.sh"
    if [[ ! -f "$POOL/pooled.csv" ]]; then
      mkdir -p "$POOL.tmp"
      cp "$PREREG_ROOT/pool/pooled.csv" "$POOL.tmp/pooled.csv"
      mv "$POOL.tmp" "$POOL"
    fi
    ;;
  fit)
    : "${N_SHARDS:?set by submit_rsa_chain_models.sh}"
    cd "$CODE_ROOT/harness_repo"
    echo "[rsa chain models] shard $SLURM_ARRAY_TASK_ID of $N_SHARDS, code $(cat "$CODE_ROOT/code_commit")"
    "$VENV_PY" -m scripts.paper.rsa_chain_models --mode fit "${COMMON[@]}" --pooled "$POOL/pooled.csv" \
      --shard "$SLURM_ARRAY_TASK_ID" --shards "$N_SHARDS" --workers "${SLURM_CPUS_PER_TASK:-8}"
    ;;
  score)
    cd "$CODE_ROOT/harness_repo"
    "$VENV_PY" -m scripts.paper.rsa_chain_models --mode score "${COMMON[@]}" \
      --test "$PREREG_ROOT/cells/prereg/run/experiment1/data/modelled.csv" \
      --out "$D/prereg/chain_models_lpd.json"
    ;;
  *) echo "ERROR: STAGE must be setup, fit or score, not $STAGE" >&2; exit 1 ;;
esac
echo "[rsa chain models $STAGE] done"
