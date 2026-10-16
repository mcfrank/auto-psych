#!/bin/bash
#SBATCH --job-name=rsa_cons_driver
#SBATCH --partition=mcfrank
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#
# The consolidation's backward elimination (docs/auto_rsa/CONSOLIDATION.md) as one Slurm array per
# round, spread over mcfrank and owners: the same rule, folds, data and fit settings as
# consolidate.sbatch STAGE=eliminate, the same on-disk layout ($WORK_ROOT/eliminate), so every fit
# that one already made is reused.
#
#   STAGE=driver (the default; submit this): replay the rule over the saved CVs
#     (`eliminate --mode plan`). When models are missing, submit them as an array of workers
#     (one model's 5-fold CV per task, on $WORKER_PARTITION with --requeue) and queue the next
#     driver after the array. When none is missing, write elimination.json and lean/ and stop.
#     A round whose models are still missing after MAX_ATTEMPTS arrays stops with an error.
#   STAGE=worker (submitted by the driver): `eliminate --mode worker --task $SLURM_ARRAY_TASK_ID`.
#
# Preemption on owners costs only the fold fits in flight: each finished fit is cached and each
# finished model's CV saved. The code is staged once ($WORK_ROOT/code_array); every round runs on it.
#
#   sbatch scripts/rsa/slurm/consolidate_array.sbatch
set -euo pipefail
export REPO="${REPO:-${SLURM_SUBMIT_DIR:-$HOME/auto-psych}}"
export WORK_ROOT="${WORK_ROOT:-${SCRATCH:?}/auto-psych/rsa_consolidate}"
PREREG_ROOT="${PREREG_ROOT:-$SCRATCH/auto-psych/rsa_prereg}"
source "$REPO/scripts/rsa/slurm/_env.sh"
STAGE="${STAGE:-driver}"
[[ "$STAGE" =~ ^(driver|worker)$ ]] || { echo "ERROR: STAGE must be driver or worker, not $STAGE" >&2; exit 1; }
WORKER_PARTITION="${WORKER_PARTITION:-mcfrank,owners}"
WORKER_CPUS="${WORKER_CPUS:-5}"        # one per fold: a model's 5 fits side by side
WORKER_MEM="${WORKER_MEM:-32GB}"       # within mcfrank's 8 GB per CPU
WORKER_TIME="${WORKER_TIME:-24:00:00}" # owners' QOS caps at 48 h
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"
CODE_ROOT="$WORK_ROOT/code_array"
POOL="$WORK_ROOT/pool"
OUT="$WORK_ROOT/eliminate"
mkdir -p "$WORK_ROOT/logs"
ELIM=(-m src.rsa.consolidate.eliminate --data "$POOL/pooled.csv" --out "$OUT" --cache "$POOL/.fit_cache")

if [[ "$STAGE" == "worker" ]]; then
  [[ -d "$CODE_ROOT/harness_repo" ]] || { echo "ERROR: no staged code at $CODE_ROOT (the driver stages it)" >&2; exit 1; }
  cd "$CODE_ROOT/harness_repo"
  exec "$VENV_PY" "${ELIM[@]}" --mode worker --task "${SLURM_ARRAY_TASK_ID:?run as an array task}" \
    --workers "${SLURM_CPUS_PER_TASK:-$WORKER_CPUS}"
fi

# --- driver -------------------------------------------------------------------------------
mkdir -p "$CODE_ROOT"
WORK_ROOT="$CODE_ROOT" HARNESS_REPO="$CODE_ROOT/harness_repo" AGENT_SRC="$CODE_ROOT/agent_src" \
  bash "$REPO/scripts/rsa/slurm/stage_code.sh"
if [[ ! -f "$POOL/pooled.csv" ]]; then
  mkdir -p "$POOL.tmp"
  cp "$PREREG_ROOT/pool/pooled.csv" "$POOL.tmp/pooled.csv"
  rsync -a "$PREREG_ROOT/pool/.fit_cache/" "$POOL.tmp/.fit_cache/"
  mv "$POOL.tmp" "$POOL"
fi
cd "$CODE_ROOT/harness_repo"
"$VENV_PY" "${ELIM[@]}" --mode plan
read -r STATUS N ATTEMPT < <("$VENV_PY" -c 'import json, sys; d = json.load(open(sys.argv[1])); print(d["status"], len(d.get("sets", [])), d.get("attempt", 0))' "$OUT/next.json")
if [[ "$STATUS" == "done" ]]; then
  echo "[driver] elimination done: $OUT/elimination.json, $OUT/lean/. Next: STAGE=score sbatch scripts/rsa/slurm/consolidate.sbatch"
  exit 0
fi
if (( ATTEMPT > MAX_ATTEMPTS )); then
  echo "FATAL: this round's models are still missing after $MAX_ATTEMPTS arrays (see $OUT/next.json and the worker logs)" >&2
  exit 1
fi
SELF="$CODE_ROOT/harness_repo/scripts/rsa/slurm/consolidate_array.sbatch"
ARRAY=$(sbatch --parsable --job-name=rsa_cons_worker --array="0-$((N - 1))" --partition="$WORKER_PARTITION" --requeue \
  --cpus-per-task="$WORKER_CPUS" --mem="$WORKER_MEM" --time="$WORKER_TIME" \
  -o "$WORK_ROOT/logs/%x_%A_%a.out" --export=ALL,STAGE=worker,REPO="$REPO",WORK_ROOT="$WORK_ROOT" "$SELF")
NEXT=$(sbatch --parsable --dependency="afterany:$ARRAY" -o "$WORK_ROOT/logs/%x_%j.out" \
  --export=ALL,STAGE=driver,REPO="$REPO",WORK_ROOT="$WORK_ROOT" "$SELF")
echo "[driver] round needs $N models (attempt $ATTEMPT): array $ARRAY on $WORKER_PARTITION; next driver $NEXT"
