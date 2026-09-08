#!/bin/bash
# Start a recovery-improvement campaign: a chain of Slurm jobs in which a
# Claude Code agent reviews the latest holdout-recovery sweeps, prescribes and
# implements improvements to the model-recovery loop on its own branch, and
# launches a sweep to test them — then, when that sweep finishes, a fresh
# review job picks up the results and iterates. Runs for days unattended.
#
# Usage:
#   bash scripts/recovery_improvement/start_campaign.sh <campaign_name> [baseline_root ...]
#
#   <campaign_name>   letters/digits/_/- ; the campaign lives at
#                     $SCRATCH/auto-psych/recovery_improvement/<campaign_name>
#   [baseline_root]   finished sweep roots the first review looks at. The FIRST
#                     is the reference every later sweep is compared against.
#                     Default: the four most recent faithful sweeps (32eig_32random
#                     first — it matches the current config's design).
#
# Knobs (env vars):
#   MAX_ITERATIONS=3            review sessions (each is followed by one sweep)
#   REVIEW_MODEL=claude-fable-5-1
#   REVIEW_MAX_TURNS=400        cap on the agent's tool calls per session
#   REVIEW_MAX_BUDGET_USD=100   cap on the session's *estimated* cost (under a
#                               subscription login this is a size cap, not billing)
#   REVIEW_TIMEOUT_SEC=21600    kill the agent after this (6h); the job's
#                               walltime (REVIEW_TIME=08:00:00) must exceed it
#   MAX_REVIEW_REPAIRS=1        re-prompt once if the deliverables are invalid
#   REVIEW_PARTITION=normal     REVIEW_CPUS=4  REVIEW_MEM=16GB
#   SWEEP_N_REPEATS=5 SWEEP_BASE_SEED=100 SWEEP_MAX_PARALLEL=5
#   SWEEP_GT_MODELS="falk_konold_dp motif_stack finite_experience_occurrence local_representativeness"
#   SWEEP_CONFIG=scripts/subjective_randomness/configs/holdout_recovery_faithful.yaml
#   AFTER_JOB=<jobid>           don't start until this job has finished (e.g. a
#                               sweep still running that should be the baseline)
#   DRY_RUN=1                   do every check and write campaign.env, but print
#                               the sbatch command instead of submitting
#   ALLOW_DIRTY=1               start even with uncommitted tracked changes
#                               (they will NOT reach the agent — it clones HEAD)
set -euo pipefail

usage() { sed -n '2,34p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
[[ $# -ge 1 ]] || usage
NAME="$1"; shift
[[ "$NAME" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "ERROR: bad campaign name '$NAME'" >&2; exit 1; }

DRIVER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_REPO="$(cd "$DRIVER_DIR/../.." && pwd)"
cd "$SOURCE_REPO"

# --- the code the agent starts from: HEAD of the current branch -------------
BASE_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
BASE_COMMIT="$(git rev-parse HEAD)"
if [[ -n "$(git status --porcelain --untracked-files=no)" && -z "${ALLOW_DIRTY:-}" ]]; then
  echo "ERROR: uncommitted changes to tracked files. The agent clones HEAD, so they" >&2
  echo "       would not reach it. Commit them, or set ALLOW_DIRTY=1 to ignore." >&2
  git status --short --untracked-files=no >&2
  exit 1
fi

# --- baselines ---------------------------------------------------------------
SCRATCH_BASE="${SCRATCH:-$GROUP_SCRATCH}/auto-psych"
if [[ $# -gt 0 ]]; then
  BASELINE_ROOTS=("$@")
else
  BASELINE_ROOTS=(
    "$SCRATCH_BASE/holdout_faithful_32eig_32random"
    "$SCRATCH_BASE/holdout_faithful_64eig"
    "$SCRATCH_BASE/holdout_faithful_64random"
    "$SCRATCH_BASE/holdout_faithful_test_retest_v8"
  )
fi
for root in "${BASELINE_ROOTS[@]}"; do
  [[ -d "$root" ]] || { echo "ERROR: baseline root does not exist: $root" >&2; exit 1; }
done
[[ -f "${BASELINE_ROOTS[0]}/test_retest.json" ]] \
  || { echo "ERROR: reference baseline has no test_retest.json: ${BASELINE_ROOTS[0]}" >&2; exit 1; }

# --- campaign root -----------------------------------------------------------
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-$SCRATCH_BASE/recovery_improvement/$NAME}"
if [[ -e "$CAMPAIGN_ROOT" ]]; then
  echo "ERROR: $CAMPAIGN_ROOT already exists; pick another name (campaigns are never reused)." >&2
  exit 1
fi
mkdir -p "$CAMPAIGN_ROOT/slurm_logs"

# --- preflight: credentials the chain needs ----------------------------------
CLAUDE_BIN_DIR=""
if command -v claude >/dev/null 2>&1; then CLAUDE_BIN_DIR="$(dirname "$(command -v claude)")"; fi
[[ -f "$HOME/.claude/.credentials.json" ]] \
  || echo "WARNING: ~/.claude/.credentials.json not found — the review job's claude will not be logged in." >&2
if ! grep -qs "^GOOGLE_API_KEY=" "$SOURCE_REPO/.secrets" 2>/dev/null; then
  echo "WARNING: no GOOGLE_API_KEY in .secrets — the sweeps' opencode+Gemini agents will fail." >&2
fi

# --- pin everything in campaign.env ------------------------------------------
REVIEW_TIME="${REVIEW_TIME:-08:00:00}"
cat > "$CAMPAIGN_ROOT/campaign.env" <<ENV
# Written by start_campaign.sh on $(date '+%Y-%m-%d %H:%M:%S'); read by every review job.
CAMPAIGN_NAME=$NAME
SOURCE_REPO=$SOURCE_REPO
BASE_BRANCH=$BASE_BRANCH
BASE_COMMIT=$BASE_COMMIT
BASELINE_ROOTS="${BASELINE_ROOTS[*]}"
MAX_ITERATIONS=${MAX_ITERATIONS:-3}
REVIEW_MODEL=${REVIEW_MODEL:-claude-fable-5-1}
REVIEW_MAX_TURNS=${REVIEW_MAX_TURNS:-400}
REVIEW_MAX_BUDGET_USD=${REVIEW_MAX_BUDGET_USD:-100}
REVIEW_TIMEOUT_SEC=${REVIEW_TIMEOUT_SEC:-21600}
MAX_REVIEW_REPAIRS=${MAX_REVIEW_REPAIRS:-1}
REVIEW_PARTITION=${REVIEW_PARTITION:-normal}
REVIEW_TIME=$REVIEW_TIME
REVIEW_CPUS=${REVIEW_CPUS:-4}
REVIEW_MEM=${REVIEW_MEM:-16GB}
DRIVER_SBATCH=$DRIVER_DIR/review_iteration.sbatch
CLAUDE_BIN_DIR=$CLAUDE_BIN_DIR
SWEEP_N_REPEATS=${SWEEP_N_REPEATS:-5}
SWEEP_BASE_SEED=${SWEEP_BASE_SEED:-100}
SWEEP_MAX_PARALLEL=${SWEEP_MAX_PARALLEL:-5}
SWEEP_GT_MODELS="${SWEEP_GT_MODELS:-falk_konold_dp motif_stack finite_experience_occurrence local_representativeness}"
SWEEP_CONFIG=${SWEEP_CONFIG:-scripts/subjective_randomness/configs/holdout_recovery_faithful.yaml}
SWEEP_SEED_MODELS_REL=${SWEEP_SEED_MODELS_REL:-src/subjective_randomness/pymc_model_families}
ENV
set -a; source "$CAMPAIGN_ROOT/campaign.env"; set +a

# --- submit iteration 1 ------------------------------------------------------
# (The chained iterations are submitted by src/recovery_improvement/slurm.py
# with the same flags — keep the two in step.)
SBATCH_ARGS=(
  ${AFTER_JOB:+--dependency=afterany:$AFTER_JOB}
  --job-name="recovery_review_$NAME"
  --partition="$REVIEW_PARTITION" --time="$REVIEW_TIME"
  --cpus-per-task="$REVIEW_CPUS" --mem="$REVIEW_MEM"
  --output="$CAMPAIGN_ROOT/slurm_logs/review_iter1_%j.out"
  --error="$CAMPAIGN_ROOT/slurm_logs/review_iter1_%j.out"
  --export=ALL,CAMPAIGN_ROOT="$CAMPAIGN_ROOT",ITERATION=1,MODE=review
  "$DRIVER_SBATCH"
)
if [[ -n "${DRY_RUN:-}" ]]; then
  echo "DRY RUN — wrote $CAMPAIGN_ROOT/campaign.env; would submit:"
  echo "  sbatch --parsable ${SBATCH_ARGS[*]}"
  exit 0
fi
job_id=$(sbatch --parsable "${SBATCH_ARGS[@]}")
echo "$job_id" > "$CAMPAIGN_ROOT/first_review_job.txt"

cat <<MSG
Started campaign '$NAME'
  root:        $CAMPAIGN_ROOT
  base:        $BASE_BRANCH @ ${BASE_COMMIT:0:10}
  baselines:   ${BASELINE_ROOTS[*]}
  iterations:  $MAX_ITERATIONS   model: $REVIEW_MODEL   turns/session: $REVIEW_MAX_TURNS   budget/session: \$$REVIEW_MAX_BUDGET_USD
  first job:   $job_id${AFTER_JOB:+ (after $AFTER_JOB)}

watch:   squeue --me ;  bash $DRIVER_DIR/campaign_status.sh $NAME
follow:  tail -f $CAMPAIGN_ROOT/slurm_logs/review_iter1_${job_id}.out
read:    $CAMPAIGN_ROOT/journal.md  and  $CAMPAIGN_ROOT/iter<N>/prescription.md
stop:    bash $DRIVER_DIR/stop_campaign.sh $NAME
MSG
