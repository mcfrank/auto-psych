#!/bin/bash
#SBATCH --job-name=rsa_open_oracle
#SBATCH --partition=mcfrank
#SBATCH --cpus-per-task=24
#SBATCH --mem=96GB
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --no-requeue
#
# How close a rehearsal's loop came to its hidden ground truth (PI 2026-10-26;
# src/rsa_open/oracle.py): the truth's theory fitted and cross-validated on the loop's own corpus and
# folds, beside every theory the loop ended with or pruned (their fits come from the loop's cache, so
# only the truth is fitted). No agents, no API spend. Default: run B' (the replay of run B's
# experiment 1).
#
#   sbatch --chdir="$HOME/auto-psych" -o "$SCRATCH/auto-psych/rsa_open_replay_b1/logs/%x_%j.out" \
#     scripts/rsa/slurm/open_oracle.sbatch
#
# Bring back $WORK_ROOT/private/oracle_<arm>_chain_<k>_exp<n>/{oracle.json,oracle.md} and the job log
# (aggregates only). Each cell and experiment has its own folder, so checks run side by side (run C: two
# cells' checks shared one and collided); not requeued, since a preempted check restarts from scratch.
#
# Overrides: WORK_ROOT (the rehearsal's root), ARM (open), EXPERIMENT (1), CHAIN (0), CV_FOLDS (5).
set -euo pipefail
export REPO="${REPO:-${SLURM_SUBMIT_DIR:-$HOME/auto-psych}}"
export WORK_ROOT="${WORK_ROOT:-${SCRATCH:?}/auto-psych/rsa_open_replay_b1}"
source "$REPO/scripts/rsa/slurm/_env.sh"
(cd "$REPO" && "$VENV_PY" -m src.rsa_open.env_check) || {
  echo "FATAL: the venv cannot run the open campaign; on a login node: source scripts/rsa/slurm/_env.sh &&" \
       "source scripts/_venv.sh && build_shared_venv" >&2; exit 1; }
CHAIN="${CHAIN:-0}"
TREE_ID="$WORK_ROOT/cells/${ARM:-open}_chain_$CHAIN/agent_tree_id"
[[ -s "$TREE_ID" ]] || { echo "ERROR: $TREE_ID is missing" >&2; exit 1; }
LOOP_DIR="$AGENT_TREES_ROOT/$(cat "$TREE_ID")/repo/_runs/chain_$CHAIN/experiment${EXPERIMENT:-1}/loop"
[[ -f "$LOOP_DIR/export.json" && -f "$LOOP_DIR/corpus.json" ]] || { echo "ERROR: $LOOP_DIR is not a finished loop" >&2; exit 1; }
GT_DIR="$WORK_ROOT/private/ground_truth"
OUT="$WORK_ROOT/private/oracle_${ARM:-open}_chain_${CHAIN}_exp${EXPERIMENT:-1}"
PYTENSOR_BASE="${L_SCRATCH:-$WORK_ROOT/.cache}/pytensor/job_${SLURM_JOB_ID:-local}"
mkdir -p "$PYTENSOR_BASE"
export PYTENSOR_FLAGS="base_compiledir=$PYTENSOR_BASE"
echo "[oracle] loop $LOOP_DIR; truth $GT_DIR; out $OUT; code $(git -C "$REPO" rev-parse HEAD 2>/dev/null)"
cd "$REPO"
"$VENV_PY" -m src.rsa_open.oracle --loop-dir "$LOOP_DIR" --truth-dir "$GT_DIR" --out "$OUT" --cv-folds "${CV_FOLDS:-5}"
echo "[oracle] done: $OUT/oracle.md"
