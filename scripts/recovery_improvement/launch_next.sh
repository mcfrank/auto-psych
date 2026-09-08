#!/bin/bash
# Launch the recovery sweep that a review iteration prepared — YOUR decision,
# after reading its prescription. Nothing in the campaign submits a sweep by
# itself unless AUTO_LAUNCH=1 was set when it was created.
#
#   bash scripts/recovery_improvement/launch_next.sh <campaign_name|campaign_root> [iteration]
#
# Defaults to the latest iteration. Reads <iter>/next_run.env (already
# validated by the review job), merges it over the campaign's sweep defaults,
# and runs the iteration repo's own submit_holdout_test_retest.sh so the
# agent's edits to the Slurm scripts/configs take effect. Output lands in
# <iter>/sweep/. Asks for confirmation unless YES=1.
#
# When the sweep has finished:  bash scripts/recovery_improvement/review.sh <campaign_name>
set -euo pipefail
[[ $# -ge 1 ]] || { echo "usage: $0 <campaign_name|campaign_root> [iteration]" >&2; exit 1; }
target="$1"
if [[ -d "$target" ]]; then CAMPAIGN_ROOT="$target"; else CAMPAIGN_ROOT="${SCRATCH:-$GROUP_SCRATCH}/auto-psych/recovery_improvement/$target"; fi
[[ -f "$CAMPAIGN_ROOT/campaign.env" ]] || { echo "ERROR: no campaign at $CAMPAIGN_ROOT" >&2; exit 1; }
set -a; source "$CAMPAIGN_ROOT/campaign.env"; set +a

if [[ $# -ge 2 ]]; then N="$2"; else
  N=$(ls -d "$CAMPAIGN_ROOT"/iter[0-9]* 2>/dev/null | sed 's/.*iter//' | sort -n | tail -1 || true)
  [[ -n "$N" ]] || { echo "ERROR: no iteration under $CAMPAIGN_ROOT yet (run review.sh first)" >&2; exit 1; }
fi
ITER_DIR="$CAMPAIGN_ROOT/iter$N"
REPO="$ITER_DIR/repo"
WORK_ROOT="$ITER_DIR/sweep"

# --- the iteration must have finished its review with a sweep decision --------
[[ -d "$ITER_DIR" ]] || { echo "ERROR: $ITER_DIR does not exist" >&2; exit 1; }
[[ -f "$ITER_DIR/STOP" ]] && { echo "ERROR: iteration $N decided STOP: $(cat "$ITER_DIR/STOP")" >&2; exit 1; }
[[ -f "$ITER_DIR/next_run.env" ]] || { echo "ERROR: $ITER_DIR/next_run.env missing — the review has not finished (or failed; see slurm_logs/)" >&2; exit 1; }
[[ -s "$ITER_DIR/prescription.md" ]] || { echo "ERROR: $ITER_DIR/prescription.md missing or empty" >&2; exit 1; }
[[ -f "$ITER_DIR/jobs.json" ]] || { echo "ERROR: $ITER_DIR/jobs.json missing — the review job did not validate the deliverables" >&2; exit 1; }
[[ -e "$WORK_ROOT" ]] && { echo "ERROR: $WORK_ROOT already exists — iteration $N's sweep was already launched (see $ITER_DIR/sweep_submit.out)" >&2; exit 1; }
[[ -d "$REPO/.git" ]] || { echo "ERROR: $REPO is not a git repo" >&2; exit 1; }
if [[ -n "$(cd "$REPO" && git status --porcelain)" ]]; then
  echo "ERROR: $REPO has uncommitted changes; the sweep copies the working tree, so it would be unreproducible:" >&2
  (cd "$REPO" && git status --short) >&2; exit 1
fi
[[ -f "$REPO/scripts/subjective_randomness/slurm/submit_holdout_test_retest.sh" ]] \
  || { echo "ERROR: submit script missing from $REPO" >&2; exit 1; }

# --- sweep env: campaign defaults, then the iteration's overrides -------------
export N_REPEATS="$SWEEP_N_REPEATS" BASE_SEED="$SWEEP_BASE_SEED" MAX_PARALLEL="$SWEEP_MAX_PARALLEL"
export GT_MODELS="$SWEEP_GT_MODELS" CONFIG="$SWEEP_CONFIG" SEED_MODELS_REL="$SWEEP_SEED_MODELS_REL"
NOTE=""
while IFS= read -r line || [[ -n "$line" ]]; do
  line="${line#"${line%%[![:space:]]*}"}"                # ltrim
  [[ -z "$line" || "$line" == \#* ]] && continue
  key="${line%%=*}"; value="${line#*=}"
  value="${value#"${value%%[![:space:]]*}"}"; value="${value%"${value##*[![:space:]]}"}"
  if [[ ${#value} -ge 2 && "${value:0:1}" == "${value: -1}" && ( "${value:0:1}" == '"' || "${value:0:1}" == "'" ) ]]; then
    value="${value:1:${#value}-2}"
  fi
  if [[ "$key" == "NOTE" ]]; then NOTE="$value"; continue; fi
  export "$key=$value"
done < "$ITER_DIR/next_run.env"
export REPO WORK_ROOT
export SBATCH_KILL_INVALID_DEP=yes   # a failed setup job kills the chain instead of leaving it pending

echo "Launch the sweep prepared by iteration $N of campaign '$CAMPAIGN_NAME'?"
echo "  note:        ${NOTE:-(none)}"
echo "  repo:        $REPO  ($(cd "$REPO" && git rev-parse --abbrev-ref HEAD) @ $(cd "$REPO" && git rev-parse --short HEAD))"
echo "  output:      $WORK_ROOT"
echo "  GT_MODELS:   $GT_MODELS"
echo "  N_REPEATS=$N_REPEATS BASE_SEED=$BASE_SEED MAX_PARALLEL=$MAX_PARALLEL CONFIG=$CONFIG"
overrides=$(grep -v '^\s*#' "$ITER_DIR/next_run.env" | grep -v '^\s*NOTE=' | grep . | tr '\n' ' ' || true)
echo "  overrides:   ${overrides:-(none — campaign defaults)}"
[[ -n "${SMOKE:-}" ]] && echo "  SMOKE MODE:  one cheap task only"
if [[ -z "${YES:-}" ]]; then
  read -r -p "Submit? [y/N] " answer
  [[ "$answer" == "y" || "$answer" == "Y" ]] || { echo "not submitted."; exit 1; }
fi

bash "$REPO/scripts/subjective_randomness/slurm/submit_holdout_test_retest.sh" | tee "$ITER_DIR/sweep_submit.out"

{
  echo "- sweep launched by $USER on $(date '+%Y-%m-%d %H:%M UTC' -u): $(grep -o 'submitted .* job: *[0-9]*' "$ITER_DIR/sweep_submit.out" | tr '\n' ';') output \`$WORK_ROOT\`"
  echo
} >> "$CAMPAIGN_ROOT/journal.md"

echo
echo "sweep running; watch with: squeue --me ;  bash $(dirname "$0")/campaign_status.sh $CAMPAIGN_NAME"
echo "when it has finished:      bash $(dirname "$0")/review.sh $CAMPAIGN_NAME"
