#!/bin/bash
#SBATCH --job-name=sr_ablation
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#
# Ablating the sr-oct26 finals (docs/sr_ablation.md): backward elimination over each final's
# mechanisms by CV grouped by stimulus pair, one Slurm array per round. Submit with
# submit_ablation.sh, which runs:
#
#   STAGE=setup   build the shared venv, stage the code once ($WORK_ROOT/code, its identity in
#                 $WORK_ROOT/code_commit; a later setup on other code refuses) and write all runs'
#                 pooled data ($WORK_ROOT/data/responses.csv) from the checkout.
#   STAGE=driver  (FINAL=run1|run2|run3) replay the rule over the saved fold fits
#                 (`eliminate --mode plan`). When fits are missing, submit them as an array of
#                 workers (one (model, fold) fit per task) and queue the next driver after it.
#                 When none is missing, write elimination.json, summary.md and lean/ and stop.
#                 A round whose fits are still missing after MAX_ATTEMPTS arrays stops with an error.
#   STAGE=worker  (submitted by the driver) `eliminate --mode worker --task $SLURM_ARRAY_TASK_ID`.
#
# Every job runs the staged code, so the checkout may move on. A preempted or requeued worker
# loses only the attempt in flight: each attempt's record is saved when it ends.
#
# Knobs (environment): REPO, WORK_ROOT, SEEDS (space-separated sampler seeds, default the
# module's), WORKER_PARTITION, WORKER_CPUS, WORKER_MEM, WORKER_TIME, MAX_PARALLEL, MAX_ATTEMPTS.
set -euo pipefail
export REPO="${REPO:-$HOME/auto-psych}"
export WORK_ROOT="${WORK_ROOT:-${SCRATCH:?}/auto-psych/sr_ablation}"
STAGE="${STAGE:-driver}"
[[ "$STAGE" =~ ^(setup|driver|worker)$ ]] || { echo "ERROR: STAGE must be setup, driver or worker, not $STAGE" >&2; exit 1; }
source "$REPO/scripts/_venv.sh"
export UV_PROJECT_ENVIRONMENT="$(shared_venv_path)"
source "$REPO/scripts/subjective_randomness/slurm/_env.sh"
CODE="$WORK_ROOT/code"
DATA="$WORK_ROOT/data/responses.csv"
WORKER_PARTITION="${WORKER_PARTITION:-normal,owners}"
WORKER_CPUS="${WORKER_CPUS:-4}"         # one per chain
WORKER_MEM="${WORKER_MEM:-32GB}"
WORKER_TIME="${WORKER_TIME:-48:00:00}"  # normal and owners cap at 48 h; two attempts of 22 h each fit
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"
mkdir -p "$WORK_ROOT/logs"
# Node-local: each process's PyTensor compile dir (a shared compile lock stalls tasks) and the
# temporary fit files (~1-2 GB each).
LOCAL="${L_SCRATCH:-/tmp}/sr_ablation_${SLURM_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$LOCAL/tmp"
export PYTENSOR_FLAGS="base_compiledir=$LOCAL/pytensor" TMPDIR="$LOCAL/tmp"

if [[ "$STAGE" == "setup" ]]; then
  build_shared_venv
  repo_code=$(bash "$REPO/scripts/subjective_randomness/slurm/code_commit.sh" "$REPO")
  if [[ -f "$WORK_ROOT/code_commit" ]]; then
    staged=$(cat "$WORK_ROOT/code_commit")
    [[ "$staged" == "$repo_code" ]] || {
      echo "ERROR: $WORK_ROOT was staged from code $staged, but $REPO is now at $repo_code." >&2
      echo "       Its fits must not continue on other code. Check out $staged, or use a new WORK_ROOT." >&2
      exit 1; }
    echo "[setup] code already staged ($staged)"
  else
    rsync -a --delete --exclude '.git' --exclude '.venv' --exclude '__pycache__' --exclude '*.nc' \
      --exclude 'node_modules' --exclude '/data/' --exclude '.secrets*' --exclude '.uv_cache' \
      --exclude '.pip_cache' --exclude '.cache' --exclude '.hf' "$REPO"/ "$CODE"/
    touch "$CODE/.here"  # pyprojroot sentinel (.git is not copied)
    printf '%s\n' "$repo_code" > "$WORK_ROOT/code_commit.tmp" && mv "$WORK_ROOT/code_commit.tmp" "$WORK_ROOT/code_commit"
    echo "[setup] staged code $repo_code"
  fi
  # The runs' data live in the checkout (data/ is not staged); an existing file must be the same.
  "$VENV_PY" -m src.subjective_randomness.ablation.pool --out "$DATA"
  exit 0
fi

: "${FINAL:?set FINAL=run1, run2 or run3 (submit_ablation.sh does)}"
[[ -f "$WORK_ROOT/code_commit" && -d "$CODE/src" ]] || { echo "ERROR: no staged code in $CODE (STAGE=setup stages it)" >&2; exit 1; }
cd "$CODE"
read -r -a seeds <<< "${SEEDS:-}"
ELIM=("$VENV_PY" -m src.subjective_randomness.ablation.eliminate --final "$FINAL" --data "$DATA"
      --out "$WORK_ROOT/$FINAL" ${seeds[@]+--seeds "${seeds[@]}"})

if [[ "$STAGE" == "worker" ]]; then
  exec "${ELIM[@]}" --mode worker --task "${SLURM_ARRAY_TASK_ID:?run as an array task}"
fi

# --- driver -------------------------------------------------------------------------------
echo "[driver $FINAL] code $(cat "$WORK_ROOT/code_commit")"
"${ELIM[@]}" --mode plan
NEXT_JSON="$WORK_ROOT/$FINAL/next.json"
read -r STATUS N ATTEMPT < <("$VENV_PY" -c 'import json, sys; d = json.load(open(sys.argv[1])); print(d["status"], len(d.get("tasks", [])), d.get("attempt", 0))' "$NEXT_JSON")
if [[ "$STATUS" == "done" ]]; then
  echo "[driver $FINAL] elimination done: $WORK_ROOT/$FINAL/elimination.json, summary.md, lean/"
  exit 0
fi
if (( ATTEMPT > MAX_ATTEMPTS )); then
  echo "FATAL: this round's fits are still missing after $MAX_ATTEMPTS arrays (see $NEXT_JSON and $WORK_ROOT/logs/sr_abl_${FINAL}_worker_*)" >&2
  exit 1
fi
SELF="$CODE/scripts/subjective_randomness/slurm/ablation.sbatch"
ARRAY=$(sbatch --parsable --job-name="sr_abl_${FINAL}_worker" --array="0-$((N - 1))${MAX_PARALLEL:+%$MAX_PARALLEL}" \
  --partition="$WORKER_PARTITION" --requeue --cpus-per-task="$WORKER_CPUS" --mem="$WORKER_MEM" --time="$WORKER_TIME" \
  -o "$WORK_ROOT/logs/%x_%A_%a.out" --export=ALL,STAGE=worker,FINAL="$FINAL",REPO="$REPO",WORK_ROOT="$WORK_ROOT" "$SELF")
NEXT=$(sbatch --parsable --job-name="sr_abl_${FINAL}_driver" --dependency="afterany:$ARRAY" -o "$WORK_ROOT/logs/%x_%j.out" \
  --export=ALL,STAGE=driver,FINAL="$FINAL",REPO="$REPO",WORK_ROOT="$WORK_ROOT" "$SELF")
echo "[driver $FINAL] round needs $N fits (attempt $ATTEMPT): array $ARRAY on $WORKER_PARTITION; next driver $NEXT"
