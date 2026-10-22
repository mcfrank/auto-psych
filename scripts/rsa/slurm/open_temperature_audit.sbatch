#!/bin/bash
#SBATCH --job-name=rsa_open_temperature_audit
#SBATCH --partition=mcfrank
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#
# What fixing E's link temperature at 1 costs each bar theory (PI 2026-10-22;
# src/rsa_open/temperature_audit.py): all 15 bar theories fitted on E (run 2's
# promote_work/all_trials.csv) with the temperature fixed and free, 30 fits at once, one core
# each, each limited to TIME_LIMIT_HOURS (a free fit that does not finish is part of the
# finding). No agents, no API spend. Runs alongside the dry run.
#
#   mkdir -p "$SCRATCH/auto-psych/rsa_open_audit/logs"
#   sbatch --chdir="$HOME/auto-psych" -o "$SCRATCH/auto-psych/rsa_open_audit/logs/%x_%j.out" \
#     scripts/rsa/slurm/open_temperature_audit.sbatch
#
# Bring back $WORK_ROOT/temperature_audit/{audit.json,audit.md} and the job log to
# data/rsa_open/temperature_audit/ (aggregates only).
#
# Overrides: TIME_LIMIT_HOURS (6), RUN2_ROOT (run 2's sweep root).
set -euo pipefail
export REPO="${REPO:-${SLURM_SUBMIT_DIR:-$HOME/auto-psych}}"
export WORK_ROOT="${WORK_ROOT:-${SCRATCH:?}/auto-psych/rsa_open_audit}"
source "$REPO/scripts/rsa/slurm/_env.sh"
(cd "$REPO" && "$VENV_PY" -m src.rsa_open.env_check) || {
  echo "FATAL: the venv cannot run the open campaign; on a login node: source scripts/rsa/slurm/_env.sh &&" \
       "source scripts/_venv.sh && build_shared_venv" >&2; exit 1; }
RUN2_ROOT="${RUN2_ROOT:-$(rsa_default_work_root)}"
EXISTING="$RUN2_ROOT/promote_work/all_trials.csv"
[[ -f "$EXISTING" ]] || { echo "ERROR: $EXISTING is missing (run promote.sbatch)" >&2; exit 1; }
OUT="$WORK_ROOT/temperature_audit"
mkdir -p "$OUT"
git -C "$REPO" rev-parse HEAD > "$OUT/code_commit" 2>/dev/null || true
PYTENSOR_BASE="${L_SCRATCH:-$WORK_ROOT/.cache}/pytensor/job_${SLURM_JOB_ID:-local}"
mkdir -p "$PYTENSOR_BASE"
export PYTENSOR_FLAGS="base_compiledir=$PYTENSOR_BASE"
echo "[audit] E = $EXISTING; code $(cat "$OUT/code_commit" 2>/dev/null); out $OUT"
cd "$REPO"
"$VENV_PY" -m src.rsa_open.temperature_audit --bar "$REPO/data/rsa_open/seeds/bar" --literature "$EXISTING" \
  --out "$OUT" --time-limit-hours "${TIME_LIMIT_HOURS:-6}"
echo "[audit] done: $OUT/audit.md"
