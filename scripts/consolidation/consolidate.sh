#!/bin/bash
# Submit the consolidation job: a Claude Code agent (Opus 4.6 by default) executes
# docs/consolidation_plan_2026_09.md phase by phase in its own clone on a
# compute node, one session per phase, requeueing itself when it must wait.
# After the consolidation (P0-P8), the raw-only refactor and agent-tree
# isolation (P9-P10) and their smoke (P11-P12), it launches the 5-repeat
# recovery sweep (P13), submits the RMSE evaluation (P14) and writes
# RESULTS.md (P15).
#
# Usage (login node; bash only, no Python):
#   bash scripts/consolidation/consolidate.sh
#
# Knobs (env vars; pinned into $WORK_ROOT/consolidation.env on the first call):
#   WORK_ROOT=$SCRATCH/auto-psych/consolidation_2026_09
#   MODEL=claude-opus-4-6          the agent model (must be usable by this login)
#   MAX_TURNS=600                  tool calls per session
#   MAX_BUDGET_USD=150             estimated-cost cap per session (a size cap
#                                  under a subscription login, not billing)
#   TIMEOUT_SEC=21600              kill a session after this (6 h)
#   PARTITION=normal TIME=2-00:00:00 CPUS=4 MEM=16GB
#   SWEEP_N_REPEATS=5 SWEEP_BASE_SEED=100 SWEEP_MAX_PARALLEL=5   (the P13 sweep)
#   SWEEP_ARMS=raw                 obsolete since the raw-only refactor (P9); kept for old env files
#   RESUME=1                       resubmit an existing work root (its clone,
#                                  progress markers and venv are kept)
#   DRY_RUN=1                      run every check, write consolidation.env,
#                                  print the sbatch command, submit nothing
#   ALLOW_DIRTY=1                  start with uncommitted tracked changes (the
#                                  agent merges HEAD, so they will NOT reach it)
set -euo pipefail

DRIVER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_REPO="$(cd "$DRIVER_DIR/../.." && pwd)"
cd "$SOURCE_REPO"
SCRATCH_BASE="${SCRATCH:-$GROUP_SCRATCH}/auto-psych"
WORK_ROOT="${WORK_ROOT:-$SCRATCH_BASE/consolidation_2026_09}"
JOB_NAME="consolidate_2026_09"

# --- frozen inputs (docs/consolidation_plan_2026_09.md §1) --------------------
CAMPAIGN_ROOT="$SCRATCH_BASE/recovery_improvement/recovery_2026_09_07"
ITER3_REPO="$CAMPAIGN_ROOT/iter3/repo"; ITER3_BRANCH="recovery-improvement/recovery_2026_09_07/iter3"
ITER4_REPO="$CAMPAIGN_ROOT/iter4/repo"; ITER4_BRANCH="recovery-improvement/recovery_2026_09_07/iter4"
ITER5_REPO="$CAMPAIGN_ROOT/iter5/repo"; ITER5_BRANCH="recovery-improvement/recovery_2026_09_07/iter5"
ITER3_SHA=ba8b2de66c45305b66066545ed8cf30c5cad208f
ITER4_SHA=470e187ed2beded050c267830380d14f1e02d27c
ITER5_SHA=2d450e21f20c5748b870e479f859766fd47b5b69
ARMC_SOURCE_REF=refs/remotes/armc/raw-features
ARMC_SHA=6ed41ea6d8994473eab0b1b79959d9ec706eb340
LEAKAGE_PATCH="$CAMPAIGN_ROOT/leakage_check_extension.patch"
LEAKAGE_PATCH_SHA256=cf3be3eed6373dda6192e3729971c47fd2884afc44651e511cc686b7b0b2764c
ARMC_RUN_ROOT="$SCRATCH_BASE/holdout_raw_features"
ITER3_SWEEP="$CAMPAIGN_ROOT/iter3/sweep"
ITER2_SWEEP="$CAMPAIGN_ROOT/iter2/sweep"                    # 5 repeats, seeds 101-105
BASELINE_SWEEP="$SCRATCH_BASE/holdout_faithful_32eig_32random"  # pre-campaign, 5 repeats

fail() { echo "ERROR: $*" >&2; exit 1; }

# A consolidation job already queued or running? Never stack them.
if squeue --me -h -n "$JOB_NAME" -o "%i" | grep -q .; then
  squeue --me -n "$JOB_NAME" >&2
  fail "a $JOB_NAME job is already queued/running"
fi

if [[ -f "$WORK_ROOT/consolidation.env" ]]; then
  [[ -n "${RESUME:-}" ]] || fail "$WORK_ROOT already has a consolidation.env; set RESUME=1 to resubmit it, or choose another WORK_ROOT"
  echo ">>> resubmitting existing work root $WORK_ROOT"
else
  # ----- preflight: every frozen input must be exactly what the plan froze ----
  [[ -f docs/consolidation_plan_2026_09.md ]] || fail "docs/consolidation_plan_2026_09.md not found"
  if [[ -n "$(git status --porcelain --untracked-files=no)" && -z "${ALLOW_DIRTY:-}" ]]; then
    git status --short --untracked-files=no >&2
    fail "uncommitted changes to tracked files; commit them (the agent merges HEAD) or set ALLOW_DIRTY=1"
  fi
  MAIN_SHA="$(git rev-parse HEAD)"
  for pair in "$ITER3_REPO:$ITER3_SHA" "$ITER4_REPO:$ITER4_SHA" "$ITER5_REPO:$ITER5_SHA"; do
    repo="${pair%%:*}"; want="${pair##*:}"
    [[ -d "$repo/.git" ]] || fail "campaign repo missing: $repo"
    got="$(git --git-dir="$repo/.git" rev-parse HEAD)"
    [[ "$got" == "$want" ]] || fail "$repo HEAD is $got, expected $want"
  done
  got="$(git rev-parse --verify "$ARMC_SOURCE_REF^{commit}" 2>/dev/null || true)"
  [[ "$got" == "$ARMC_SHA" ]] || fail "$ARMC_SOURCE_REF is '$got', expected $ARMC_SHA (git fetch \$SCRATCH/auto-psych/arm_c/repo arm-c/raw-features first)"
  [[ -f "$LEAKAGE_PATCH" ]] || fail "leakage patch missing: $LEAKAGE_PATCH"
  echo "$LEAKAGE_PATCH_SHA256  $LEAKAGE_PATCH" | sha256sum --check --quiet || fail "leakage patch checksum mismatch"
  [[ -d "$ARMC_RUN_ROOT" ]] || fail "arm C run root missing: $ARMC_RUN_ROOT"
  [[ -d "$ITER3_SWEEP" ]] || fail "iteration 3 sweep missing: $ITER3_SWEEP"
  [[ -d "$ITER2_SWEEP" ]] || fail "iteration 2 sweep missing: $ITER2_SWEEP"
  [[ -d "$BASELINE_SWEEP" ]] || fail "baseline sweep missing: $BASELINE_SWEEP"
  [[ -f "$HOME/.claude/.credentials.json" ]] || fail "~/.claude/.credentials.json not found — claude is not logged in"
  CLAUDE_BIN_DIR=""
  if command -v claude >/dev/null 2>&1; then CLAUDE_BIN_DIR="$(dirname "$(command -v claude)")"; fi
  grep -qs "^GOOGLE_API_KEY=" .secrets \
    || echo "WARNING: no GOOGLE_API_KEY in .secrets — the P7 smoke cells' opencode+Gemini agents would fail." >&2

  mkdir -p "$WORK_ROOT/slurm_logs"
  cat > "$WORK_ROOT/consolidation.env" <<ENV
# Written by consolidate.sh on $(date '+%Y-%m-%d %H:%M:%S'); read by every consolidation job.
SOURCE_REPO=$SOURCE_REPO
MAIN_SHA=$MAIN_SHA
ITER3_REPO=$ITER3_REPO
ITER3_BRANCH=$ITER3_BRANCH
ITER3_SHA=$ITER3_SHA
ITER4_REPO=$ITER4_REPO
ITER4_BRANCH=$ITER4_BRANCH
ITER4_SHA=$ITER4_SHA
ITER5_REPO=$ITER5_REPO
ITER5_BRANCH=$ITER5_BRANCH
ITER5_SHA=$ITER5_SHA
ARMC_SOURCE_REF=$ARMC_SOURCE_REF
ARMC_SHA=$ARMC_SHA
LEAKAGE_PATCH=$LEAKAGE_PATCH
LEAKAGE_PATCH_SHA256=$LEAKAGE_PATCH_SHA256
ARMC_RUN_ROOT=$ARMC_RUN_ROOT
ITER3_SWEEP=$ITER3_SWEEP
ITER2_SWEEP=$ITER2_SWEEP
BASELINE_SWEEP=$BASELINE_SWEEP
SWEEP_ARMS=${SWEEP_ARMS:-raw}
SWEEP_N_REPEATS=${SWEEP_N_REPEATS:-5}
SWEEP_BASE_SEED=${SWEEP_BASE_SEED:-100}
SWEEP_MAX_PARALLEL=${SWEEP_MAX_PARALLEL:-5}
MODEL=${MODEL:-claude-opus-4-6}
MAX_TURNS=${MAX_TURNS:-600}
MAX_BUDGET_USD=${MAX_BUDGET_USD:-150}
TIMEOUT_SEC=${TIMEOUT_SEC:-21600}
JOB_NAME=$JOB_NAME
PARTITION=${PARTITION:-normal}
TIME=${TIME:-2-00:00:00}
CPUS=${CPUS:-4}
MEM=${MEM:-16GB}
DRIVER_SBATCH=$DRIVER_DIR/consolidate.sbatch
CLAUDE_BIN_DIR=$CLAUDE_BIN_DIR
UV_CACHE_DIR_SHARED=$CAMPAIGN_ROOT/.uv_cache
GIT_USER_NAME="Ben Prystawski"
GIT_USER_EMAIL=benpry@stanford.edu
ENV
  echo ">>> new work root $WORK_ROOT (main @ ${MAIN_SHA:0:10}, base iter3 @ ${ITER3_SHA:0:10})"
fi
set -a; source "$WORK_ROOT/consolidation.env"; set +a

SBATCH_ARGS=(
  --job-name="$JOB_NAME"
  --partition="$PARTITION" --time="$TIME"
  --cpus-per-task="$CPUS" --mem="$MEM"
  --output="$WORK_ROOT/slurm_logs/%x_%j.out"
  --error="$WORK_ROOT/slurm_logs/%x_%j.out"
  --mail-type=END,FAIL,TIMEOUT
  --export=ALL,WORK_ROOT="$WORK_ROOT"
  "$DRIVER_SBATCH"
)
if [[ -n "${DRY_RUN:-}" ]]; then
  echo "DRY RUN — would submit:"
  echo "  sbatch --parsable ${SBATCH_ARGS[*]}"
  exit 0
fi
job_id=$(sbatch --parsable "${SBATCH_ARGS[@]}")
echo "$job_id" >> "$WORK_ROOT/jobs.txt"
cat <<MSG
submitted consolidation job $job_id
  model $MODEL, up to $MAX_TURNS turns / ${TIMEOUT_SEC}s per phase session; phases P0..P15
  (requeues itself for session limits, walltime, the smoke jobs, the P13 sweep, the P14 evaluation)
  sweep: $SWEEP_N_REPEATS repeats, BASE_SEED=$SWEEP_BASE_SEED, $SWEEP_MAX_PARALLEL concurrent

follow:   tail -f $WORK_ROOT/slurm_logs/${JOB_NAME}_${job_id}.out
status:   cat $WORK_ROOT/STATUS.md; ls $WORK_ROOT/progress/
smoke:    cat $WORK_ROOT/VERDICT.md $WORK_ROOT/HANDOFF.md      # when P12 is done
result:   cat $WORK_ROOT/RESULTS.md                            # when P15 is done
branch:   git fetch $WORK_ROOT/repo consolidate/2026-09
stop:     scancel $job_id   (the P7 smoke chains, if submitted, are separate jobs)
MSG
