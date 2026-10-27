#!/bin/bash
#SBATCH --job-name=ng_ablation
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#
# Ablating the number-game run's final (docs/ng_ablation.md): backward elimination over its
# mechanisms by CV grouped by example set, one Slurm array per round. Modelled on the sr-oct26
# ablation's scripts/subjective_randomness/slurm/ablation.sbatch. Submit with
# submit_ablation.sh, which runs:
#
#   STAGE=setup   build the shared venv, stage the code once ($WORK_ROOT/code, its identity in
#                 $WORK_ROOT/code_commit; a later setup on other code refuses) and copy the run's
#                 responses ($WORK_ROOT/data/responses.csv) from the checkout.
#   STAGE=driver  replay the rule over the saved fold fits (`eliminate --mode plan`). When fits
#                 are missing, submit them as an array of workers (TASKS_PER_JOB (model, fold)
#                 fits per array task, one after another) and queue the next driver after it.
#                 When none is missing, write elimination.json, summary.md and lean/ and stop.
#                 A round whose fits are still missing after MAX_ATTEMPTS arrays stops with an error.
#   STAGE=worker  (submitted by the driver) `eliminate --mode worker --task $SLURM_ARRAY_TASK_ID`.
#
# Every job runs the staged code, so the checkout may move on. A preempted or requeued worker
# loses only the attempt in flight: each attempt's record is saved when it ends.
#
# Knobs (environment): REPO, WORK_ROOT, SEEDS (space-separated sampler seeds, default the
# module's), WORKER_PARTITION, WORKER_CPUS, WORKER_MEM, WORKER_TIME, TASKS_PER_JOB,
# MAX_PARALLEL, MAX_ATTEMPTS.
set -euo pipefail
export REPO="${REPO:-$HOME/auto-psych}"
export WORK_ROOT="${WORK_ROOT:-${SCRATCH:?}/auto-psych/ng_ablation}"
STAGE="${STAGE:-driver}"
[[ "$STAGE" =~ ^(setup|driver|worker)$ ]] || { echo "ERROR: STAGE must be setup, driver or worker, not $STAGE" >&2; exit 1; }
source "$REPO/scripts/_venv.sh"
export UV_PROJECT_ENVIRONMENT="$(shared_venv_path)"
source "$REPO/scripts/subjective_randomness/slurm/_env.sh"
CODE="$WORK_ROOT/code"
DATA="$WORK_ROOT/data/responses.csv"
WORKER_PARTITION="${WORKER_PARTITION:-normal,owners}"
WORKER_CPUS="${WORKER_CPUS:-4}"         # one per chain
WORKER_MEM="${WORKER_MEM:-8GB}"
WORKER_TIME="${WORKER_TIME:-08:00:00}"  # a fit took ~6 min (docs/ng_ablation.md); a task of 5 ~30 min
TASKS_PER_JOB="${TASKS_PER_JOB:-5}"     # one fit alone is under Sherlock's 10-minute minimum job
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"
mkdir -p "$WORK_ROOT/logs"
# Node-local: each process's PyTensor compile dir (a shared compile lock stalls tasks) and the
# temporary fit files.
LOCAL="${L_SCRATCH:-/tmp}/ng_ablation_${SLURM_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}"
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
  # The run's data live in the checkout (data/ is not staged); an existing file must be the same.
  "$VENV_PY" -m src.number_game.ablation.data --out "$DATA"
  exit 0
fi

[[ -f "$WORK_ROOT/code_commit" && -d "$CODE/src" ]] || { echo "ERROR: no staged code in $CODE (STAGE=setup stages it)" >&2; exit 1; }
cd "$CODE"
read -r -a seeds <<< "${SEEDS:-}"
ELIM=("$VENV_PY" -m src.number_game.ablation.eliminate --final ng --data "$DATA"
      --out "$WORK_ROOT/ng" --tasks-per-job "$TASKS_PER_JOB" ${seeds[@]+--seeds "${seeds[@]}"})

if [[ "$STAGE" == "worker" ]]; then
  exec "${ELIM[@]}" --mode worker --task "${SLURM_ARRAY_TASK_ID:?run as an array task}"
fi

# --- driver -------------------------------------------------------------------------------
echo "[driver] code $(cat "$WORK_ROOT/code_commit")"
"${ELIM[@]}" --mode plan
NEXT_JSON="$WORK_ROOT/ng/next.json"
read -r STATUS N ATTEMPT < <("$VENV_PY" -c 'import json, sys; d = json.load(open(sys.argv[1])); print(d["status"], len(d.get("tasks", [])), d.get("attempt", 0))' "$NEXT_JSON")
if [[ "$STATUS" == "done" ]]; then
  echo "[driver] elimination done: $WORK_ROOT/ng/elimination.json, summary.md, lean/"
  exit 0
fi
if (( ATTEMPT > MAX_ATTEMPTS )); then
  echo "FATAL: this round's fits are still missing after $MAX_ATTEMPTS arrays (see $NEXT_JSON and $WORK_ROOT/logs/ng_abl_worker_*)" >&2
  exit 1
fi
JOBS=$(( (N + TASKS_PER_JOB - 1) / TASKS_PER_JOB ))
SELF="$CODE/scripts/number_game/slurm/ablation.sbatch"
ARRAY=$(sbatch --parsable --job-name=ng_abl_worker --array="0-$((JOBS - 1))${MAX_PARALLEL:+%$MAX_PARALLEL}" \
  --partition="$WORKER_PARTITION" --requeue --cpus-per-task="$WORKER_CPUS" --mem="$WORKER_MEM" --time="$WORKER_TIME" \
  -o "$WORK_ROOT/logs/%x_%A_%a.out" --export=ALL,STAGE=worker,REPO="$REPO",WORK_ROOT="$WORK_ROOT",TASKS_PER_JOB="$TASKS_PER_JOB" "$SELF")
NEXT=$(sbatch --parsable --job-name=ng_abl_driver --dependency="afterany:$ARRAY" -o "$WORK_ROOT/logs/%x_%j.out" \
  --export=ALL,STAGE=driver,REPO="$REPO",WORK_ROOT="$WORK_ROOT",TASKS_PER_JOB="$TASKS_PER_JOB" "$SELF")
echo "[driver] round needs $N fits in $JOBS array tasks (attempt $ATTEMPT): array $ARRAY on $WORKER_PARTITION; next driver $NEXT"
