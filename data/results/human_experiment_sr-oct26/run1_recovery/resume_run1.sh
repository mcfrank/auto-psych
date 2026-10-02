#!/bin/bash
# ============================================================================
#  Resume full live run 1 on the fit-cache fix (2026-09-30).
#
#  Run 1 (job 45982167) collected experiment 1 (40 people, Prolific study
#  <study id>) and was in experiment 1's model loop, which ran
#  without a fit cache and re-sampled every model at every admission. This
#  script replaces that job with two:
#
#    job A  experiment 1's model loop only (RESUME_AGENTS=5_model_loop): the
#           stage restarts from its recorded input; no deploy, no study.
#    job B  experiments 2-3 (EXPERIMENTS=2-3): design, implement, deploy and
#           TWO NEW PROLIFIC STUDIES (real recruiting, real money). Starts only
#           if job A succeeds (--dependency=afterok).
#
#  Usage (in order):
#    bash resume_run1.sh check    read-only preflight; changes nothing
#    bash resume_run1.sh cancel   scancel the old run-1 job (discards its model loop)
#    bash resume_run1.sh submit   once the old job is gone: sync the fixed code
#                                 into run 1's copy and submit jobs A and B
#
#  NEVER relaunch run 1 with start_full_run.sh: it deletes $WORK_ROOT/run1.
# ============================================================================
set -euo pipefail

OLD_JOB=45982167
EXPECTED_COMMIT=e6c67591709c5fffb175582736b3bcaaaf913f55
CONFIG_NAME=full_run_sr-oct26.yaml
RUN=run1
EXPECTED_SITE=auto-psych-2c5da-0926-run1
EXPECTED_STUDY=<study id>
EXPECTED_ROWS=2560
OWNER=kushin            # experiment 1's recorded collection_owner
TIME_A=1-12:00:00       # one model loop (estimated 5-8 h with the cache); default QOS: long needs >= 48 h
TIME_B=3-00:00:00       # two experiments: design + recruiting (<= 3 h each) + model loops

MODE="${1:-}"
case "$MODE" in check|cancel|submit) ;; *) echo "usage: bash $0 check|cancel|submit" >&2; exit 2 ;; esac

die() { echo "STOP: $*" >&2; exit 1; }
[[ "$EXPECTED_COMMIT" != "__FILL__" ]] || die "EXPECTED_COMMIT is not filled in"

source "$HOME/repos/live_env.sh"          # REPO, OUTER_LIVE_SLURM_DIR
source "$OUTER_LIVE_SLURM_DIR/_env.sh"    # WORK_ROOT, VENV_PY, modules, secrets
source "$OUTER_LIVE_SLURM_DIR/_hosting_site.sh"

WT="$WORK_ROOT/runs/$RUN/repo"
OUT="$WORK_ROOT/$RUN/data"
EXP="$OUT/subjective_randomness"
LOGDIR="$WORK_ROOT/slurm_logs"
RSYNC_EXCLUDES=(
  --exclude '.git' --exclude '.secrets' --exclude '.venv' --exclude 'data'
  --exclude '__pycache__' --exclude '*.nc' --exclude 'scratch' --exclude '.worktrees'
  --exclude 'public' --exclude 'firebase.generated.json' --exclude 'functions/node_modules'
  --exclude '.uv_cache' --exclude '.pip_cache' --exclude '.cache' --exclude '.hf'
)   # exactly submit_parallel.sh's: public/ (experiment 1's live page) etc. stay

old_job_state() { squeue -h -j "$OLD_JOB" -o %T 2>/dev/null || true; }

preflight() {
  echo "--- code"
  cd "$REPO"
  local head; head="$(git rev-parse HEAD)"
  [[ "$head" == "$EXPECTED_COMMIT" ]] || die "checkout $REPO is at $head, expected $EXPECTED_COMMIT"
  [[ -z "$(git status --porcelain)" ]] || die "checkout $REPO has uncommitted changes"
  grep -q 'cache_dir=exp_dir / "model_loop" / ".fit_cache"' src/pipelines/outer_loop/run.py \
    || die "the fit-cache fix is not in $REPO/src/pipelines/outer_loop/run.py"
  echo "  checkout $REPO at $head, clean, fix present"

  echo "--- run 1's tree"
  local rows; rows=$(( $(wc -l < "$EXP/experiment1/data/responses.csv") - 1 ))
  [[ "$rows" -eq "$EXPECTED_ROWS" ]] || die "experiment 1 has $rows response rows, expected $EXPECTED_ROWS"
  local recorded
  recorded="$("$VENV_PY" -c 'import json,sys; m=json.load(open(sys.argv[1])); print(m["prolific_study_id"], m["hosting_site"])' \
    "$EXP/experiment1/deployment/deployment_manifest.json")"
  [[ "$recorded" == "$EXPECTED_STUDY $EXPECTED_SITE" ]] \
    || die "experiment 1's manifest records '$recorded', expected '$EXPECTED_STUDY $EXPECTED_SITE'"
  [[ -d "$EXP/experiment1/cognitive_models_input" ]] || die "experiment 1 has no recorded model-loop input"
  [[ ! -e "$EXP/experiment2" ]] || die "$EXP/experiment2 already exists"
  [[ -d "$WT/public/e1-run1" ]] || die "run copy $WT has no public/e1-run1 (experiment 1's page)"
  echo "  experiment 1: $rows rows, study $EXPECTED_STUDY, site $EXPECTED_SITE; no experiment 2 yet"

  echo "--- config $CONFIG_NAME (Prolific token check and cost summary below)"
  local cfg_env
  cfg_env="$("$VENV_PY" "$OUTER_LIVE_SLURM_DIR/_pilot_config.py" "$REPO/$CONFIG_NAME" --check)" \
    || die "config / Prolific token validation failed"
  eval "$cfg_env"
  [[ "$N_PARTICIPANTS" == 40 && "$PROLIFIC_MODE" == live && "$CONFIRM_LIVE_RECRUITMENT" == 1 \
     && "$CODING_AGENT" == claude && "$CLAUDE_AUTH" == api && "$CODING_AGENT_MODEL" == claude-opus-5-5 \
     && "$INNER_LOOP_ITERATIONS" == 5 && "$INNER_LOOP_CANDIDATES" == 6 && "$QOS" == long ]] \
    || die "config settings differ from run 1's"
  SERIES_LABEL="$RUN_LABEL"
  SITE="$(hosting_site "$FIREBASE_PROJECT" "$SERIES_LABEL" "$RUN")"
  [[ "$SITE" == "$EXPECTED_SITE" ]] || die "config gives site $SITE, run 1 used $EXPECTED_SITE"
  EXPORT_COMMON="ALL,RUN_LABEL=$RUN,RUN_WORKTREE=$WT,CODING_AGENT=$CODING_AGENT,AUTO_PSYCH_OUTPUT_DIR=$OUT,AUTO_PSYCH_HOSTING_SITE=$SITE,AUTO_PSYCH_COLLECTION_OWNER=$OWNER"
  unset EXPERIMENTS EXPERIMENT RESUME_AGENTS
  echo "  N=$N_PARTICIPANTS mode=$PROLIFIC_MODE agent=$CODING_AGENT/$CODING_AGENT_MODEL ($CLAUDE_AUTH) rounds=$INNER_LOOP_ITERATIONS x $INNER_LOOP_CANDIDATES site=$SITE"
}

sync_dry_run() {
  rsync -a --delete --dry-run --itemize-changes "${RSYNC_EXCLUDES[@]}" "$REPO"/ "$WT"/ | grep -v '^\.d' || true
}

case "$MODE" in
  check)
    preflight
    echo "--- old job $OLD_JOB: $(old_job_state || true)"
    echo "--- code the submit would change in $WT (dry run):"
    sync_dry_run
    echo "--- would submit:"
    echo "  A: sbatch --time=$TIME_A EXPERIMENT=1 RESUME_AGENTS=5_model_loop run_live.sbatch"
    echo "  B: sbatch --time=$TIME_B --qos=$QOS --dependency=afterok:<A> EXPERIMENTS=2-3 run_live.sbatch"
    echo "  env: $EXPORT_COMMON"
    ;;
  cancel)
    state="$(old_job_state)"
    [[ -n "$state" ]] || { echo "job $OLD_JOB is not in the queue (already ended); next: bash $0 submit"; exit 0; }
    scontrol show job "$OLD_JOB" | grep -qE "JobName=outer_live_run1( |$)" || die "job $OLD_JOB is not outer_live_run1"
    scancel "$OLD_JOB"
    echo "scancel sent to $OLD_JOB (was $state). Give it a minute, then: bash $0 submit"
    ;;
  submit)
    [[ -z "$(old_job_state)" ]] || die "old job $OLD_JOB is still in the queue ($(old_job_state)); cancel it and wait for it to end"
    preflight
    echo "--- syncing $REPO into $WT (changes:)"
    sync_dry_run
    rsync -a --delete "${RSYNC_EXCLUDES[@]}" "$REPO"/ "$WT"/
    touch "$WT/.here"
    (cd "$REPO" && "$VENV_PY" -m src.pipelines.outer_loop.deployment.record_provenance --checkout "$REPO" --copy "$WT")
    prov="$("$VENV_PY" -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p["git_commit"], p["git_dirty"])' "$WT/code_provenance.json")"
    [[ "$prov" == "$EXPECTED_COMMIT False" ]] || die "copy provenance is '$prov'"
    cmp -s "$REPO/src/pipelines/outer_loop/run.py" "$WT/src/pipelines/outer_loop/run.py" || die "copy's run.py differs from the checkout's"
    [[ -d "$WT/public/e1-run1" ]] || die "the sync removed public/e1-run1"
    echo "  copy at $prov"

    ja=$(sbatch --parsable --job-name="outer_live_${RUN}_A" \
      --output="$LOGDIR/%x_%j.out" --error="$LOGDIR/%x_%j.out" \
      --time="$TIME_A" \
      --export="$EXPORT_COMMON,EXPERIMENT=1,RESUME_AGENTS=5_model_loop" \
      "$OUTER_LIVE_SLURM_DIR/run_live.sbatch")
    echo "submitted A (experiment 1 model loop): job $ja"
    jb=$(sbatch --parsable --job-name="outer_live_${RUN}_B" \
      --output="$LOGDIR/%x_%j.out" --error="$LOGDIR/%x_%j.out" \
      --time="$TIME_B" --qos="$QOS" --dependency="afterok:$ja" \
      --export="$EXPORT_COMMON,EXPERIMENTS=2-3" \
      "$OUTER_LIVE_SLURM_DIR/run_live.sbatch")
    echo "submitted B (experiments 2-3, publishes 2 studies): job $jb, after A succeeds"
    echo
    echo "logs: $LOGDIR/outer_live_${RUN}_A_$ja.out  $LOGDIR/outer_live_${RUN}_B_$jb.out"
    echo "If A fails, B stays pending (DependencyNeverSatisfied): scancel $jb."
    echo "TO STOP a published study: pause it in the Prolific dashboard, then scancel."
    ;;
esac
