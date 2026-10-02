#!/bin/bash
# ============================================================================
#  Queue full live run 1's experiments 2-3 on EXPECTED_COMMIT (2026-09-30):
#  the novelty-prediction fix, the design over person-level models (new
#  participants) and pruning at 4·dse, after experiment 1 was re-pruned at 4
#  (reprune_run1.sbatch MODE=apply).
#
#  Job A (experiment 1's model loop) runs from run 1's copy, which cannot be
#  updated until A ends. This script snapshots the clean checkout beside the
#  copy (runs/run1/repo_next) and submits run1_exps23.sbatch, which starts
#  after A succeeds, syncs the snapshot into the copy and runs experiments 2-3
#  there: TWO NEW PROLIFIC STUDIES (real recruiting, real money).
#
#    bash queue_run1_exps23.sh check    read-only preflight; changes nothing
#    bash queue_run1_exps23.sh submit   make the snapshot and submit the job
# ============================================================================
set -euo pipefail

JOB_A=46034338
EXPECTED_COMMIT=7db302b60151e4997d851b969ba8a00f5b230da4
CONFIG_NAME=full_run_sr-oct26.yaml
RUN=run1
EXPECTED_SITE=auto-psych-2c5da-0926-run1
EXPECTED_STUDY=<study id>
EXPECTED_ROWS=2560
OWNER=kushin            # experiment 1's recorded collection_owner
TIME_B=5-00:00:00       # two experiments (design, <= 3 h recruiting, model loop each); long QOS

MODE="${1:-}"
case "$MODE" in check|submit) ;; *) echo "usage: bash $0 check|submit" >&2; exit 2 ;; esac
die() { echo "STOP: $*" >&2; exit 1; }
[[ "$EXPECTED_COMMIT" != "__FILL__" ]] || die "EXPECTED_COMMIT is not filled in"

source "$HOME/repos/live_env.sh"          # REPO, OUTER_LIVE_SLURM_DIR
source "$OUTER_LIVE_SLURM_DIR/_env.sh"    # WORK_ROOT, VENV_PY, modules, secrets
source "$OUTER_LIVE_SLURM_DIR/_hosting_site.sh"

WT="$WORK_ROOT/runs/$RUN/repo"
SNAP="$WORK_ROOT/runs/$RUN/repo_next"
OUT="$WORK_ROOT/$RUN/data"
EXP="$OUT/subjective_randomness"
LOGDIR="$WORK_ROOT/slurm_logs"
WRAPPER="$WORK_ROOT/run1_exps23.sbatch"
RSYNC_EXCLUDES=(
  --exclude '.git' --exclude '.secrets' --exclude '.venv' --exclude 'data'
  --exclude '__pycache__' --exclude '*.nc' --exclude 'scratch' --exclude '.worktrees'
  --exclude 'public' --exclude 'firebase.generated.json' --exclude 'functions/node_modules'
  --exclude '.uv_cache' --exclude '.pip_cache' --exclude '.cache' --exclude '.hf'
)   # exactly submit_parallel.sh's
prov() { "$VENV_PY" -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p["git_commit"], p["git_dirty"])' "$1"; }

preflight() {
  echo "--- code"
  cd "$REPO"
  local head; head="$(git rev-parse HEAD)"
  [[ "$head" == "$EXPECTED_COMMIT" ]] || die "checkout $REPO is at $head, expected $EXPECTED_COMMIT"
  [[ -z "$(git status --porcelain)" ]] || die "checkout $REPO has uncommitted changes"
  grep -q 'cache_dir=exp_dir / "model_loop" / ".fit_cache"' src/pipelines/outer_loop/run.py \
    || die "the fit-cache fix is not in the checkout"
  grep -q 'novelty_predictions=novelty_predictions' src/pipelines/inner_loop/pymc_orchestrator.py \
    || die "the novelty-prediction fix is not in the checkout"
  grep -q 'def _new_participant_draws' src/pipelines/outer_loop/eig.py \
    || die "the design over person-level models is not in the checkout"
  [[ -f "$WRAPPER" ]] || die "no $WRAPPER"
  echo "  checkout at $head, clean, both fixes present"

  echo "--- run 1's tree"
  local rows; rows=$(( $(wc -l < "$EXP/experiment1/data/responses.csv") - 1 ))
  [[ "$rows" -eq "$EXPECTED_ROWS" ]] || die "experiment 1 has $rows response rows, expected $EXPECTED_ROWS"
  local recorded
  recorded="$("$VENV_PY" -c 'import json,sys; m=json.load(open(sys.argv[1])); print(m["prolific_study_id"], m["hosting_site"])' \
    "$EXP/experiment1/deployment/deployment_manifest.json")"
  [[ "$recorded" == "$EXPECTED_STUDY $EXPECTED_SITE" ]] \
    || die "experiment 1's manifest records '$recorded', expected '$EXPECTED_STUDY $EXPECTED_SITE'"
  [[ ! -e "$EXP/experiment2" ]] || die "$EXP/experiment2 already exists"
  local carried
  carried="$("$VENV_PY" -c 'import json,sys; r=json.load(open(sys.argv[1]))[-1]; print(r["dse_multiplier"], len(r["live"]))' \
    "$EXP/experiment1/model_loop/repruned.json")" || die "experiment 1 was not re-pruned (no repruned.json)"
  [[ "$carried" == "4.0 "* && "${carried#4.0 }" -ge 2 ]] || die "experiment 1's re-prune record reads '$carried'"
  "$VENV_PY" -c 'import sys; from src.pipelines.outer_loop.orchestrator_validators import validate_cc_output as v; ok, m = v("5_model_loop", __import__("pathlib").Path(sys.argv[1])); sys.exit(0 if ok else m)' \
    "$EXP/experiment1" || die "experiment 1's model-loop stage does not validate"
  [[ -d "$WT/public/e1-run1" ]] || die "run copy $WT has no public/e1-run1"
  echo "  experiment 1: $rows rows, study $EXPECTED_STUDY, site $EXPECTED_SITE, re-pruned at 4 carrying ${carried#4.0 } models; no experiment 2 yet"

  echo "--- job A ($JOB_A)"
  [[ -z "$(squeue --me -h -n "outer_live_${RUN}_B" -o %i)" ]] || die "an outer_live_${RUN}_B job is already queued"
  # From the queue listing: squeue -j on a purged (finished) job prints an error, not a state.
  local state; state="$(squeue --me -h -o "%i %T" | awk -v j="$JOB_A" '$1 == j {print $2}')"
  if [[ -n "$state" ]]; then
    DEPENDENCY="afterok:$JOB_A"
    echo "  $state: experiments 2-3 will start only if it succeeds"
  else
    state="$(sacct -n -X -j "$JOB_A" -o State%20 | tr -d ' ')"
    [[ "$state" == COMPLETED && -f "$EXP/experiment1/model_loop/export_complete.json" ]] \
      || die "job A ended as '$state' (or left no export record): find the cause first"
    DEPENDENCY=""
    echo "  COMPLETED with experiment 1's export record: experiments 2-3 can start now"
  fi

  echo "--- config $CONFIG_NAME (Prolific token check and cost summary below)"
  local cfg_env
  cfg_env="$("$VENV_PY" "$OUTER_LIVE_SLURM_DIR/_pilot_config.py" "$REPO/$CONFIG_NAME" --check)" \
    || die "config / Prolific token validation failed"
  eval "$cfg_env"
  [[ "$N_PARTICIPANTS" == 40 && "$PROLIFIC_MODE" == live && "$CONFIRM_LIVE_RECRUITMENT" == 1 \
     && "$CODING_AGENT" == claude && "$CLAUDE_AUTH" == api && "$CODING_AGENT_MODEL" == claude-opus-5-5 \
     && "$INNER_LOOP_ITERATIONS" == 5 && "$INNER_LOOP_CANDIDATES" == 6 && "$QOS" == long \
     && "$PRUNE_DSE_MULTIPLIER" == 4 ]] \
    || die "config settings differ from run 1's"
  SITE="$(hosting_site "$FIREBASE_PROJECT" "$RUN_LABEL" "$RUN")"
  [[ "$SITE" == "$EXPECTED_SITE" ]] || die "config gives site $SITE, run 1 used $EXPECTED_SITE"
  unset EXPERIMENTS EXPERIMENT RESUME_AGENTS
  EXPORT="ALL,RUN_LABEL=$RUN,RUN_WORKTREE=$WT,CODING_AGENT=$CODING_AGENT,AUTO_PSYCH_OUTPUT_DIR=$OUT,AUTO_PSYCH_HOSTING_SITE=$SITE,AUTO_PSYCH_COLLECTION_OWNER=$OWNER,EXPERIMENTS=2-3,EXPECTED_COMMIT=$EXPECTED_COMMIT"
  SBATCH_ARGS=(--job-name="outer_live_${RUN}_B" --output="$LOGDIR/%x_%j.out" --error="$LOGDIR/%x_%j.out"
               --time="$TIME_B" --qos="$QOS" ${DEPENDENCY:+--dependency="$DEPENDENCY"})
  echo "  N=$N_PARTICIPANTS mode=$PROLIFIC_MODE agent=$CODING_AGENT/$CODING_AGENT_MODEL ($CLAUDE_AUTH) rounds=$INNER_LOOP_ITERATIONS x $INNER_LOOP_CANDIDATES prune=${PRUNE_DSE_MULTIPLIER}·dse site=$SITE"
}

case "$MODE" in
  check)
    preflight
    echo "--- the request (sbatch --test-only; submits nothing):"
    sbatch --test-only --time="$TIME_B" --qos="$QOS" "$WRAPPER" 2>&1 | tail -1
    echo "--- would snapshot $REPO into $SNAP, then submit:"
    echo "  sbatch ${SBATCH_ARGS[*]} --export=$EXPORT $WRAPPER"
    ;;
  submit)
    preflight
    echo "--- snapshot $REPO -> $SNAP"
    mkdir -p "$SNAP"
    rsync -a --delete "${RSYNC_EXCLUDES[@]}" "$REPO"/ "$SNAP"/
    touch "$SNAP/.here"
    (cd "$REPO" && "$VENV_PY" -m src.pipelines.outer_loop.deployment.record_provenance --checkout "$REPO" --copy "$SNAP")
    [[ "$(prov "$SNAP/code_provenance.json")" == "$EXPECTED_COMMIT False" ]] || die "snapshot provenance is '$(prov "$SNAP/code_provenance.json")'"
    echo "  snapshot at $EXPECTED_COMMIT"
    jb=$(sbatch --parsable "${SBATCH_ARGS[@]}" --export="$EXPORT" "$WRAPPER")
    echo "submitted run 1 experiments 2-3: job $jb (${DEPENDENCY:-no dependency})"
    echo "log: $LOGDIR/outer_live_${RUN}_B_$jb.out"
    echo "If job A fails, this job stays pending (DependencyNeverSatisfied): scancel $jb."
    echo "TO STOP a published study: pause it in the Prolific dashboard, then scancel."
    ;;
esac
