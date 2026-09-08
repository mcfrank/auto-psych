#!/bin/bash
# Submit ONE review job: a Claude Code agent reads the most recent
# holdout-recovery sweeps, writes a prescription for improving the model-
# recovery loop, implements one change on its own branch, and writes the spec
# of the sweep that would test it (next_run.env). It does NOT launch that
# sweep — you read the prescription and run launch_next.sh if you want it.
# When that sweep has finished, run this script again: the next review picks
# up its results.
#
# Usage:
#   bash scripts/recovery_improvement/review.sh <campaign_name> [baseline_root ...]
#
#   <campaign_name>   letters/digits/_/- ; the campaign lives at
#                     $SCRATCH/auto-psych/recovery_improvement/<campaign_name>.
#                     A new name creates the campaign (iteration 1); an existing
#                     one adds the next iteration.
#   [baseline_root]   (first call only) finished sweep roots the first review
#                     looks at; the FIRST is the reference every later sweep is
#                     compared against. Default: the four most recent faithful
#                     sweeps, holdout_faithful_32eig_32random first (it matches
#                     the current config's design).
#
# Knobs (env vars; pinned into campaign.env on the first call):
#   REVIEW_MODEL=claude-fable-5-1
#   REVIEW_MAX_TURNS=400        cap on the agent's tool calls per session
#   REVIEW_MAX_BUDGET_USD=100   cap on the session's *estimated* cost (under a
#                               subscription login this is a size cap, not billing)
#   REVIEW_TIMEOUT_SEC=21600    kill the agent after this (6h); the job's
#                               walltime (REVIEW_TIME=08:00:00) must exceed it
#   MAX_REVIEW_REPAIRS=1        re-prompt once if the deliverables are invalid
#   REVIEW_PARTITION=normal     REVIEW_CPUS=4  REVIEW_MEM=16GB
#   MAX_ITERATIONS=20           hard cap on iterations per campaign
#   AUTO_LAUNCH=0               1 = each review launches its sweep and chains
#                               the next review itself (unattended campaign)
#   SWEEP_N_REPEATS=5 SWEEP_BASE_SEED=100 SWEEP_MAX_PARALLEL=5
#   SWEEP_GT_MODELS="falk_konold_dp motif_stack finite_experience_occurrence local_representativeness"
#   SWEEP_CONFIG=scripts/subjective_randomness/configs/holdout_recovery_faithful.yaml
#   AFTER_JOB=<jobid>           don't start until this job has finished
#   DRY_RUN=1                   do every check, write campaign.env, print the
#                               sbatch command instead of submitting
#   ALLOW_DIRTY=1               start even with uncommitted tracked changes
#                               (they will NOT reach the agent — it clones HEAD)
set -euo pipefail

usage() { sed -n '2,44p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
[[ $# -ge 1 ]] || usage
NAME="$1"; shift
[[ "$NAME" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "ERROR: bad campaign name '$NAME'" >&2; exit 1; }

DRIVER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_REPO="$(cd "$DRIVER_DIR/../.." && pwd)"
cd "$SOURCE_REPO"
SCRATCH_BASE="${SCRATCH:-$GROUP_SCRATCH}/auto-psych"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-$SCRATCH_BASE/recovery_improvement/$NAME}"

# A review job for this campaign already queued or running? Never stack them.
if squeue --me -h -n "recovery_review_$NAME" -o "%i" | grep -q .; then
  echo "ERROR: a review job for '$NAME' is already queued/running:" >&2
  squeue --me -n "recovery_review_$NAME" >&2
  exit 1
fi

if [[ -f "$CAMPAIGN_ROOT/campaign.env" ]]; then
  # ----- continue an existing campaign: next iteration ---------------------
  [[ $# -eq 0 ]] || echo "WARNING: baseline roots are pinned on the first call; ignoring: $*" >&2
  set -a; source "$CAMPAIGN_ROOT/campaign.env"; set +a
  [[ -f "$CAMPAIGN_ROOT/STOP" ]] && { echo "ERROR: campaign is stopped ($CAMPAIGN_ROOT/STOP); remove the file to continue" >&2; exit 1; }
  last=$(ls -d "$CAMPAIGN_ROOT"/iter[0-9]* 2>/dev/null | sed 's/.*iter//' | sort -n | tail -1 || true)
  last="${last:-0}"
  ITERATION=$(( last + 1 ))
  (( ITERATION <= MAX_ITERATIONS )) || { echo "ERROR: MAX_ITERATIONS=$MAX_ITERATIONS reached" >&2; exit 1; }
  if (( last >= 1 )); then
    prev="$CAMPAIGN_ROOT/iter$last"
    if [[ ! -f "$prev/prescription.md" || ( ! -f "$prev/next_run.env" && ! -f "$prev/STOP" ) ]]; then
      echo "ERROR: iteration $last has no finished deliverables (prescription.md + next_run.env|STOP)." >&2
      echo "       Its review job may still be running or have failed; see $CAMPAIGN_ROOT/slurm_logs/" >&2
      exit 1
    fi
    [[ -f "$prev/STOP" ]] && echo "NOTE: iteration $last decided STOP ($(head -c 120 "$prev/STOP")); starting iteration $ITERATION anyway." >&2
    if [[ ! -f "$prev/sweep/test_retest.json" ]]; then
      if [[ -d "$prev/sweep" ]]; then
        echo "WARNING: iteration $last's sweep has not finished ($prev/sweep has no test_retest.json)." >&2
      else
        echo "WARNING: iteration $last's sweep was never launched (no $prev/sweep). Run launch_next.sh $NAME $last first" >&2
      fi
      echo "         The new review will see whatever exists; set FORCE=1 to proceed anyway." >&2
      [[ -n "${FORCE:-}" ]] || exit 1
    fi
  fi
  echo ">>> campaign '$NAME': iteration $ITERATION (previous: $last)"
else
  # ----- new campaign: iteration 1 --------------------------------------
  [[ -e "$CAMPAIGN_ROOT" ]] && { echo "ERROR: $CAMPAIGN_ROOT exists but has no campaign.env" >&2; exit 1; }
  ITERATION=1
  BASE_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
  BASE_COMMIT="$(git rev-parse HEAD)"
  if [[ -n "$(git status --porcelain --untracked-files=no)" && -z "${ALLOW_DIRTY:-}" ]]; then
    echo "ERROR: uncommitted changes to tracked files. The agent clones HEAD, so they" >&2
    echo "       would not reach it. Commit them, or set ALLOW_DIRTY=1 to ignore." >&2
    git status --short --untracked-files=no >&2
    exit 1
  fi
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

  CLAUDE_BIN_DIR=""
  if command -v claude >/dev/null 2>&1; then CLAUDE_BIN_DIR="$(dirname "$(command -v claude)")"; fi
  [[ -f "$HOME/.claude/.credentials.json" ]] \
    || echo "WARNING: ~/.claude/.credentials.json not found — the review job's claude will not be logged in." >&2
  if ! grep -qs "^GOOGLE_API_KEY=" "$SOURCE_REPO/.secrets" 2>/dev/null; then
    echo "WARNING: no GOOGLE_API_KEY in .secrets — a launched sweep's opencode+Gemini agents would fail." >&2
  fi

  mkdir -p "$CAMPAIGN_ROOT/slurm_logs"
  cat > "$CAMPAIGN_ROOT/campaign.env" <<ENV
# Written by review.sh on $(date '+%Y-%m-%d %H:%M:%S'); read by every review job.
CAMPAIGN_NAME=$NAME
SOURCE_REPO=$SOURCE_REPO
BASE_BRANCH=$BASE_BRANCH
BASE_COMMIT=$BASE_COMMIT
BASELINE_ROOTS="${BASELINE_ROOTS[*]}"
MAX_ITERATIONS=${MAX_ITERATIONS:-20}
AUTO_LAUNCH=${AUTO_LAUNCH:-0}
REVIEW_MODEL=${REVIEW_MODEL:-claude-fable-5-1}
REVIEW_MAX_TURNS=${REVIEW_MAX_TURNS:-400}
REVIEW_MAX_BUDGET_USD=${REVIEW_MAX_BUDGET_USD:-100}
REVIEW_TIMEOUT_SEC=${REVIEW_TIMEOUT_SEC:-21600}
MAX_REVIEW_REPAIRS=${MAX_REVIEW_REPAIRS:-1}
REVIEW_PARTITION=${REVIEW_PARTITION:-normal}
REVIEW_TIME=${REVIEW_TIME:-08:00:00}
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
  # (the array is replaced by the scalar from campaign.env below; keep the list for the echo)
  baselines_shown="${BASELINE_ROOTS[*]}"
  unset BASELINE_ROOTS
  set -a; source "$CAMPAIGN_ROOT/campaign.env"; set +a
  echo ">>> new campaign '$NAME' at $CAMPAIGN_ROOT (base $BASE_BRANCH @ ${BASE_COMMIT:0:10})"
  echo "    baselines: ${baselines_shown}"
fi

# --- submit the review job -----------------------------------------------------
# (With AUTO_LAUNCH=1 the chained iterations are submitted by
# src/recovery_improvement/slurm.py with the same flags — keep the two in step.)
SBATCH_ARGS=(
  ${AFTER_JOB:+--dependency=afterany:$AFTER_JOB}
  --job-name="recovery_review_$NAME"
  --partition="$REVIEW_PARTITION" --time="$REVIEW_TIME"
  --cpus-per-task="$REVIEW_CPUS" --mem="$REVIEW_MEM"
  --output="$CAMPAIGN_ROOT/slurm_logs/review_iter${ITERATION}_%j.out"
  --error="$CAMPAIGN_ROOT/slurm_logs/review_iter${ITERATION}_%j.out"
  --export=ALL,CAMPAIGN_ROOT="$CAMPAIGN_ROOT",ITERATION="$ITERATION",MODE=review
  "$DRIVER_SBATCH"
)
if [[ -n "${DRY_RUN:-}" ]]; then
  echo "DRY RUN — would submit:"
  echo "  sbatch --parsable ${SBATCH_ARGS[*]}"
  exit 0
fi
job_id=$(sbatch --parsable "${SBATCH_ARGS[@]}")
echo "$job_id" >> "$CAMPAIGN_ROOT/review_jobs.txt"

cat <<MSG
submitted review job $job_id  (iteration $ITERATION of campaign '$NAME')${AFTER_JOB:+, after $AFTER_JOB}
  model $REVIEW_MODEL, up to $REVIEW_MAX_TURNS turns / ${REVIEW_TIMEOUT_SEC}s; auto-launch: $AUTO_LAUNCH

follow:  tail -f $CAMPAIGN_ROOT/slurm_logs/review_iter${ITERATION}_${job_id}.out
status:  bash $DRIVER_DIR/campaign_status.sh $NAME
then:    read $CAMPAIGN_ROOT/iter$ITERATION/prescription.md
         bash $DRIVER_DIR/launch_next.sh $NAME $ITERATION     # if you want the sweep it proposes
         bash $DRIVER_DIR/review.sh $NAME                     # once that sweep has finished
MSG
