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
# RESOURCES -- TO BE SET from docs/auto_rsa/SMOKE_RESULTS_2.md (branch
# auto-rsa-smoke2): its measured seed-fit and admission-fit times on the
# combined data, and the run directory's size. A cell's wall time is about
#   5 seed fits + 5 rounds x (agents <= 3 x AGENT_TIMEOUT_SEC [first try,
#   retry, repair] + up to 12 admission fits, one at a time)
#   + held-out scoring (cache hits) [+ one ground-truth fit for recovery].
# The defaults are placeholders sized like the subjective-randomness cells.
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"   # six agents' self-checks run at once
MEM="${MEM:-64G}"                      # normal allows <= 8 GB/core
TIME="${TIME:-48:00:00}"               # > 48 h adds --qos=long (max 7 days)
SETUP_CPUS="${SETUP_CPUS:-16}"         # two ground-truth fits at once
SETUP_MEM="${SETUP_MEM:-32G}"
SETUP_TIME="${SETUP_TIME:-06:00:00}"
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
