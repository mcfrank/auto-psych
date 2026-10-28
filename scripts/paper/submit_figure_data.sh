#!/bin/bash
# ============================================================================
#  Submit a paper figure-data job in parallel: the models every run removed
#  (Figure 2's "pruned during the run") scored in an array of shards, then
#  the figure data assembled from their records.
#
#    cd $HOME/auto-psych && bash scripts/paper/submit_figure_data.sh sr
#    cd $HOME/auto-psych && bash scripts/paper/submit_figure_data.sh number_game
#
#  Three jobs, chained:
#    1. setup (figure_data_setup.sbatch): builds the venv once and records
#       the checkout's commit in $FIT_WORK_DIR/code_commit;
#    2. the shards (figure_data_shard.sbatch, an array of N_SHARDS, at most
#       MAX_PARALLEL at once), each scoring its share of the removed models
#       into $FIT_WORK_DIR; a task refuses to run on another commit;
#    3. the assembly (<domain>_figure_data.sbatch), after every shard has
#       ended (afterany): it fits the rest and writes the figure data, and
#       raises naming any removed model a failed shard left unscored.
#  Resubmitting reuses every record and fit already in $FIT_WORK_DIR. Keep
#  the checkout unchanged until the assembly has finished.
#
#  Knobs (environment): N_SHARDS, MAX_PARALLEL, SHARD_CPUS, SHARD_MEM,
#  SHARD_TIME, WORK_ROOT, FIT_WORK_DIR, REPO.
# ============================================================================
set -euo pipefail
DOMAIN="${1:?usage: submit_figure_data.sh sr|number_game}"
export REPO="${REPO:-$HOME/auto-psych}"
export WORK_ROOT="${WORK_ROOT:-$SCRATCH/auto-psych/paper}"
case "$DOMAIN" in
  sr)  # 176 removed models on each run's data (one fit each at 3,000 draws x 4 chains) and
       # 254 other models on all runs pooled (23k trials, 1,500 draws after 4,000 tuning steps)
    MODULE=scripts.paper.sr_figure_data
    ASSEMBLY=scripts/paper/sr_figure_data.sbatch
    export FIT_WORK_DIR="${FIT_WORK_DIR:-$WORK_ROOT/sr_figure_data}"
    N_SHARDS="${N_SHARDS:-86}" SHARD_CPUS="${SHARD_CPUS:-16}"
    SHARD_MEM="${SHARD_MEM:-64G}" SHARD_TIME="${SHARD_TIME:-24:00:00}" ;;
  number_game)  # 170 removed models on each run's data (7,560 responses) and 250 other
                # models on all runs pooled (22,680), 5 fits each (all data + 4 CV folds)
    MODULE=scripts.paper.number_game_figure_data
    ASSEMBLY=scripts/paper/number_game_figure_data.sbatch
    export FIT_WORK_DIR="${FIT_WORK_DIR:-$WORK_ROOT/number_game_figure_data}"
    N_SHARDS="${N_SHARDS:-84}" SHARD_CPUS="${SHARD_CPUS:-16}"
    SHARD_MEM="${SHARD_MEM:-32G}" SHARD_TIME="${SHARD_TIME:-08:00:00}" ;;
  *) echo "FATAL: domain must be sr or number_game, not $DOMAIN" >&2; exit 1 ;;
esac
MAX_PARALLEL="${MAX_PARALLEL:-$N_SHARDS}"
cd "$REPO"
[[ -z "$(git status --porcelain)" ]] || echo "[paper] WARNING: the checkout is dirty; the jobs run it as it is" >&2
mkdir -p "$FIT_WORK_DIR"

setup=$(sbatch --parsable --job-name="paper_${DOMAIN}_setup" \
  --export=ALL,REPO,WORK_ROOT,FIT_WORK_DIR scripts/paper/figure_data_setup.sbatch)
shards=$(sbatch --parsable --dependency="afterok:$setup" --job-name="paper_${DOMAIN}_shard" \
  --array="0-$((N_SHARDS - 1))%$MAX_PARALLEL" \
  --cpus-per-task="$SHARD_CPUS" --mem="$SHARD_MEM" --time="$SHARD_TIME" \
  --export=ALL,REPO,WORK_ROOT,FIT_WORK_DIR,MODULE="$MODULE",N_SHARDS="$N_SHARDS" \
  scripts/paper/figure_data_shard.sbatch)
assembly=$(sbatch --parsable --dependency="afterany:$shards" \
  --export=ALL,REPO,WORK_ROOT,FIT_WORK_DIR "$ASSEMBLY")
echo "[paper] $DOMAIN: setup $setup -> $N_SHARDS shards $shards -> assembly $assembly"
echo "[paper] work dir $FIT_WORK_DIR"
