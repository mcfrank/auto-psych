#!/bin/bash
# ============================================================================
#  Submit the reference-games chain-model scoring (rsa_chain_models.sbatch):
#  setup -> an array of fit shards -> score (after every shard has ended,
#  afterany: it raises naming any model a failed shard left without a record).
#
#    cd $HOME/auto-psych && bash scripts/paper/submit_rsa_chain_models.sh
#
#  224 models (2026-10-19): 8 per shard, one per core, as the bar's fit (8
#  CPUs, 48 GB, 1-3 h for 17 models). Knobs (environment): N_SHARDS,
#  MAX_PARALLEL, SHARD_CPUS, SHARD_MEM, SHARD_TIME, WORK_ROOT, PREREG_ROOT.
#  Resubmitting reuses every record and fit already in $WORK_ROOT; keep the
#  checkout's data/rsa/ unchanged until the score job has finished.
# ============================================================================
set -euo pipefail
export REPO="${REPO:-$HOME/auto-psych}"
export WORK_ROOT="${WORK_ROOT:-$SCRATCH/auto-psych/rsa_chain_models}"
export PREREG_ROOT="${PREREG_ROOT:-$SCRATCH/auto-psych/rsa_prereg}"
N_SHARDS="${N_SHARDS:-28}"
MAX_PARALLEL="${MAX_PARALLEL:-$N_SHARDS}"
SHARD_CPUS="${SHARD_CPUS:-8}" SHARD_MEM="${SHARD_MEM:-48G}" SHARD_TIME="${SHARD_TIME:-12:00:00}"
cd "$REPO"
[[ -f "$PREREG_ROOT/pool/pooled.csv" ]] || { echo "FATAL: no pooled data at $PREREG_ROOT/pool/pooled.csv" >&2; exit 1; }
SB=scripts/paper/rsa_chain_models.sbatch
setup=$(sbatch --parsable --job-name=paper_rsa_setup --cpus-per-task=2 --mem=8G --time=00:30:00 \
  --export=ALL,REPO,WORK_ROOT,PREREG_ROOT,STAGE=setup "$SB")
fit=$(sbatch --parsable --dependency="afterok:$setup" --job-name=paper_rsa_fit \
  --array="0-$((N_SHARDS - 1))%$MAX_PARALLEL" \
  --cpus-per-task="$SHARD_CPUS" --mem="$SHARD_MEM" --time="$SHARD_TIME" \
  --export=ALL,REPO,WORK_ROOT,PREREG_ROOT,STAGE=fit,N_SHARDS="$N_SHARDS" "$SB")
score=$(sbatch --parsable --dependency="afterany:$fit" --job-name=paper_rsa_score \
  --cpus-per-task=2 --mem=16G --time=01:00:00 \
  --export=ALL,REPO,WORK_ROOT,PREREG_ROOT,STAGE=score "$SB")
echo "[paper] rsa chain models: setup $setup -> $N_SHARDS fit shards $fit -> score $score"
echo "[paper] work root $WORK_ROOT; result $REPO/data/rsa/prereg/chain_models_lpd.json"
