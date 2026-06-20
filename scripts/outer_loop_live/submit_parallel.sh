#!/bin/bash
# Launch K parallel LIVE outer-loop runs on Sherlock. Each run gets:
#   * its own git worktree         -> isolated public/, firebase.generated.json,
#                                     functions/
#   * its own --run-label          -> /e{N}-{label}/ hosting path + Firestore
#                                     collection_session_id + its own Prolific study
#   * its own AUTO_PSYCH_OUTPUT_DIR -> isolated experiment{N}/ data tree
# The brief `firebase deploy` step is serialized across runs by the shared
# AUTO_PSYCH_DEPLOY_LOCK (set in _env.sh); everything else runs concurrently.
#
# Usage:   bash scripts/outer_loop_live/submit_parallel.sh
# Knobs:   K=3 EXPERIMENT=1 N_PARTICIPANTS=20 PROJECT=subjective_randomness \
#          DESIGN_MODE=agent WORK_ROOT=$SCRATCH/auto-psych/outer_loop_live \
#            bash scripts/outer_loop_live/submit_parallel.sh
#
# Prereqs (see README.md): run setup.sbatch once; put PROLIFIC_API_TOKEN +
# FIREBASE_TOKEN in .secrets; set a real prolific_config.yaml for the project.

set -euo pipefail

# Absolute dir of this script — passed to every job so they find _env.sh.
OUTER_LIVE_SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OUTER_LIVE_SLURM_DIR
source "$OUTER_LIVE_SLURM_DIR/_env.sh"   # sets REPO, WORK_ROOT, caches, lock

K="${K:-3}"
export PROJECT="${PROJECT:-subjective_randomness}"
export EXPERIMENT="${EXPERIMENT:-1}"
export N_PARTICIPANTS="${N_PARTICIPANTS:-1}"
export FIREBASE_PROJECT="${FIREBASE_PROJECT:-auto-psych-2c5da}"
export DESIGN_MODE="${DESIGN_MODE:-agent}"

# Worktrees check out HEAD, so uncommitted edits to src/ would NOT reach them.
if [[ -n "$(git -C "$REPO" status --porcelain)" ]]; then
  echo "ERROR: $REPO has uncommitted changes; per-run worktrees check out HEAD." >&2
  echo "       Commit (or stash) your edits first so they reach the worktrees." >&2
  exit 1
fi

LOGDIR="$WORK_ROOT/slurm_logs"; mkdir -p "$LOGDIR"
WT_ROOT="$WORK_ROOT/worktrees"; mkdir -p "$WT_ROOT"
HEAD_SHA="$(git -C "$REPO" rev-parse --short HEAD)"

for i in $(seq 1 "$K"); do
  LABEL="run${i}"
  WT="$WT_ROOT/$LABEL"
  OUT="$WORK_ROOT/$LABEL/data"; mkdir -p "$OUT"
  # Detached worktree at HEAD (idempotent: reuse if a prior submit made it).
  if [[ ! -d "$WT" ]]; then
    git -C "$REPO" worktree add --detach "$WT" HEAD
  fi
  jid=$(sbatch --parsable \
    --job-name="outer_live_$LABEL" \
    --output="$LOGDIR/%x_%j.out" --error="$LOGDIR/%x_%j.out" \
    --export=ALL,RUN_LABEL="$LABEL",RUN_WORKTREE="$WT",AUTO_PSYCH_OUTPUT_DIR="$OUT" \
    "$OUTER_LIVE_SLURM_DIR/run_live.sbatch")
  echo "submitted $LABEL: job $jid  (worktree=$WT  out=$OUT)"
done

echo
echo "code @ $HEAD_SHA   |   watch: squeue --me   |   logs: $LOGDIR"
echo "Tear down worktrees when done:  git -C $REPO worktree remove <path>"
