#!/bin/bash
# The RSA sweep's cells. Sourced by rsa_loop_array.sbatch, rsa_setup.sbatch,
# rsa_status.sh and submit.sh (it loads no modules and changes no directory,
# so the status script can use it on a login node).
#
# Run 2 (PI decision 2026-10-08, from run 1's round-by-round results): real
# data x 3 replicates x 8 rounds (run 1's real cells were still improving at
# round 5, and the two replicates differed by ~90 lpd), recovery x 2
# replicates x 4 rounds (run 1's recovery cells were flat after round 3).
# Override with REAL_REPLICATES, REAL_ROUNDS, RECOVERY_REPLICATES,
# RECOVERY_ROUNDS (exported for prepare, submit and status alike).
#
# Task -> cell: the long real cells first, then recovery replicate-major, so a
# throttled array (%6) starts every long cell at once and the cell that waits
# for a slot is a short one:
#   0 real_rep1  1 real_rep2  2 real_rep3
#   3 recovery_literal_rep1  4 recovery_salience_rep1
#   5 recovery_literal_rep2  6 recovery_salience_rep2
#
# A replicate is a different loop seed on the same data and split (PI decision
# 2026-10-07). LOOP_SEED = BASE_LOOP_SEED + replicate - 1.

RSA_CONDITIONS=(real recovery_literal recovery_salience)

rsa_condition_replicates() {
  case "$1" in
    real) echo "${REAL_REPLICATES:-3}" ;;
    recovery_literal|recovery_salience) echo "${RECOVERY_REPLICATES:-2}" ;;
    *) echo "unknown RSA condition: $1" >&2; return 1 ;;
  esac
}
rsa_condition_rounds() {
  case "$1" in
    real) echo "${REAL_ROUNDS:-8}" ;;
    recovery_literal|recovery_salience) echo "${RECOVERY_ROUNDS:-4}" ;;
    *) echo "unknown RSA condition: $1" >&2; return 1 ;;
  esac
}

RSA_CELL_LIST=()
for (( _r = 1; _r <= $(rsa_condition_replicates real); _r++ )); do RSA_CELL_LIST+=("real:$_r"); done
_max_rec=$(rsa_condition_replicates recovery_literal)
for (( _r = 1; _r <= _max_rec; _r++ )); do
  for _c in recovery_literal recovery_salience; do RSA_CELL_LIST+=("$_c:$_r"); done
done
unset _r _c _max_rec
RSA_N_CELLS=${#RSA_CELL_LIST[@]}
# The seed models' directory, relative to a repo (harness copy or agent tree).
RSA_SEED_MODELS_REL="src/pipelines/outer_loop/projects/rsa_reference/seed_models"

# Ground truth and the seeds the loop starts without, per condition. The
# ground truth is always the first excluded seed; the rest are its near-twins.
rsa_condition_gt() {
  case "$1" in
    real)              echo "" ;;
    recovery_literal)  echo "literal_listener" ;;
    recovery_salience) echo "rsa_l1_salience" ;;
    *) echo "unknown RSA condition: $1" >&2; return 1 ;;
  esac
}
rsa_condition_excluded_seeds() {
  case "$1" in
    real)              echo "" ;;
    recovery_literal)  echo "literal_listener" ;;
    recovery_salience) echo "rsa_l1_salience rsa_l1_shared_prior" ;;
    *) echo "unknown RSA condition: $1" >&2; return 1 ;;
  esac
}

# rsa_cell <task>: sets CONDITION REPLICATE CELL GT EXCLUDED_SEEDS LOOP_SEED ROUNDS.
rsa_cell() {
  local task="$1"
  [[ "$task" =~ ^[0-9]+$ ]] && (( task < RSA_N_CELLS )) \
    || { echo "task $task is outside 0-$(( RSA_N_CELLS - 1 ))" >&2; return 1; }
  local entry="${RSA_CELL_LIST[$task]}"
  CONDITION="${entry%%:*}"
  REPLICATE="${entry##*:}"
  CELL="${CONDITION}_rep${REPLICATE}"
  GT="$(rsa_condition_gt "$CONDITION")"
  EXCLUDED_SEEDS="$(rsa_condition_excluded_seeds "$CONDITION")"
  LOOP_SEED=$(( ${BASE_LOOP_SEED:-0} + REPLICATE - 1 ))
  ROUNDS="$(rsa_condition_rounds "$CONDITION")"
}

# The data a condition's loop trains and is tested on, under $DATA_ROOT.
rsa_condition_data_dir() {
  local data_root="$1" condition="$2"
  echo "$data_root/$condition"
}

# Does this checkout's loop CLI take --seed? (src/rsa/loop/run.py did not before
# 2026-10-07; without it the replicates differ only by the agents' sampling.)
# Run from a repo root (the harness copy). The help is captured first: piped
# into grep -q, the CLI could die of SIGPIPE and pipefail would read a "no".
rsa_loop_cli_has_seed() {
  local python="$1" help
  help="$("$python" -m src.rsa.loop.run --help 2>&1)" || true
  grep -qE -- '(^|[[:space:]])--seed[[:space:]]' <<< "$help"
}

# The sweep's venv: $GROUP_HOME/venvs/auto-psych_<sweep> (Sherlock's rule:
# Python environments live in $GROUP_HOME, not on $SCRATCH), else
# $WORK_ROOT/venv. UV_PROJECT_ENVIRONMENT, if exported, wins.
# The sweep root every script defaults to. One place: after run 1, three
# scripts still defaulted to rsa_run1 and run 2's first submit went there.
RSA_SWEEP_NAME="rsa_run2"
rsa_default_work_root() {
  local base="${SCRATCH:-${GROUP_SCRATCH:-}}"
  [[ -n "$base" ]] || { echo "set WORK_ROOT (or SCRATCH)" >&2; return 1; }
  echo "$base/auto-psych/$RSA_SWEEP_NAME"
}

rsa_default_venv() {
  local work_root="$1"
  if [[ -n "${UV_PROJECT_ENVIRONMENT:-}" ]]; then
    echo "$UV_PROJECT_ENVIRONMENT"
  elif [[ -n "${GROUP_HOME:-}" ]]; then
    # One venv for every RSA job: run 2's, built by its prepare step. A venv named
    # after each WORK_ROOT (rehearsals, live) was never built (live stage 1, 2026-10-10).
    echo "$GROUP_HOME/venvs/auto-psych_$RSA_SWEEP_NAME"
  else
    echo "$work_root/venv"
  fi
}
