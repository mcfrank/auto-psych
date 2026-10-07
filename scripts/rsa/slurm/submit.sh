#!/bin/bash
# Submit RSA run 1 from a login node, after prepare_data.sh: the setup job,
# then the six-cell array with afterok on it.
#
#   bash scripts/rsa/slurm/submit.sh                # setup + array
#   SKIP_SETUP=1 bash scripts/rsa/slurm/submit.sh   # array only (setup done)
#   ARRAY=4 RESTART=1 SKIP_SETUP=1 bash ...         # restart one failed cell
#
# Every variable below can be overridden from the environment; Slurm exports
# the environment to the jobs (--export=ALL), so loop settings such as
# MAX_ITERATIONS, AGENT_TIMEOUT_SEC or ALLOW_SHARED_LOOP_SEED pass through.
# Logs: $WORK_ROOT/logs/rsa_setup_<job>.out, rsa_loop_<array>_<task>.out.
set -euo pipefail
export REPO="${REPO:-$HOME/auto-psych}"
export WORK_ROOT="${WORK_ROOT:-${SCRATCH:-${GROUP_SCRATCH:?set WORK_ROOT (or SCRATCH)}}/auto-psych/rsa_run1}"

# ============================================================================
# Sizes from the 2026-10-07 dry runs on the combined data (40k training
# trials; full NUTS 4 x (1000 + 1000); docs/auto_rsa/HANDOFF_sherlock_run1.md
# section 4): one fit ~50-75 s single-threaded, ~1.3 GB; a round's fits and
# scoring ~5-7 min on 4 CPUs; the loop process grows to ~5 GB over 5 rounds;
# an agent's self-check ~90 s, ~2.5 GB. Agents dominate a cell's wall time:
#   5 rounds x (agents <= 3 x AGENT_TIMEOUT_SEC [first try, retry, repair]
#   + ~5 min of fits) + ~2 min of scoring.
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"    # 6 agents' self-checks, or a round's 6-12 fits, at once
MEM="${MEM:-32G}"                      # loop ~5 GB + 6 self-checks x ~2.5 GB + agents; normal allows <= 8 GB/core
TIME="${TIME:-24:00:00}"               # worst case ~11 h (every agent to its timeout); > 48 h adds --qos=long
SETUP_CPUS="${SETUP_CPUS:-4}"          # two ground-truth fits, ~70 s each
SETUP_MEM="${SETUP_MEM:-8G}"
SETUP_TIME="${SETUP_TIME:-01:00:00}"
# ============================================================================
PARTITION="${PARTITION:-normal}"
ARRAY="${ARRAY:-0-5}"
MAX_PARALLEL="${MAX_PARALLEL:-6}"

# Whole hours in a Slurm time limit ([D-]HH[:MM[:SS]]), rounded up.
hours_of() {
  local t="$1" days=0 h m s
  if [[ "$t" == *-* ]]; then days="${t%%-*}"; t="${t#*-}"; fi
  IFS=: read -r h m s <<< "$t"
  h=$((10#${h:-0})); m=$((10#${m:-0})); s=$((10#${s:-0}))
  echo $(( days * 24 + h + ( (m > 0 || s > 0) ? 1 : 0 ) ))
}
qos_args() {
  if (( $(hours_of "$1") > 48 )); then echo "--qos=long"; fi
}

[[ -f "$WORK_ROOT/data/real/train.csv" ]] || { echo "ERROR: no prepared data in $WORK_ROOT/data; run prepare_data.sh first" >&2; exit 1; }
mkdir -p "$WORK_ROOT/logs"
cd "$WORK_ROOT/logs"
SLURM_DIR="$REPO/scripts/rsa/slurm"

echo "WORK_ROOT=$WORK_ROOT"
echo "array $ARRAY%$MAX_PARALLEL: $CPUS_PER_TASK CPUs, $MEM, $TIME $(qos_args "$TIME") on $PARTITION"
dependency=()
if [[ "${SKIP_SETUP:-}" != 1 ]]; then
  echo "setup: $SETUP_CPUS CPUs, $SETUP_MEM, $SETUP_TIME $(qos_args "$SETUP_TIME")"
  # shellcheck disable=SC2046  # qos_args prints zero or one word
  setup_id=$(sbatch --parsable --export=ALL -p "$PARTITION" -c "$SETUP_CPUS" --mem="$SETUP_MEM" \
    -t "$SETUP_TIME" $(qos_args "$SETUP_TIME") "$SLURM_DIR/rsa_setup.sbatch")
  echo "submitted setup: $setup_id"
  dependency=(--dependency="afterok:$setup_id")
fi
# shellcheck disable=SC2046
array_id=$(sbatch --parsable --export=ALL -p "$PARTITION" -c "$CPUS_PER_TASK" --mem="$MEM" \
  -t "$TIME" $(qos_args "$TIME") --array="$ARRAY%$MAX_PARALLEL" \
  ${dependency[@]+"${dependency[@]}"} "$SLURM_DIR/rsa_loop_array.sbatch")
echo "submitted array: $array_id"
echo "monitor: bash $SLURM_DIR/rsa_status.sh   (logs in $WORK_ROOT/logs)"
