#!/bin/bash
# ============================================================================
#  Rerun full live run 1's experiment 3 model loop at 128 GB (2026-10-01).
#
#  Job B (46172873) was OOM-killed in round 2 of experiment 3's model loop
#  (67 GB peak of 64). Experiment 3's data are collected; its model loop
#  restarts from its recorded input (RESUME_AGENTS=5_model_loop: no deploy,
#  no study). Runs from run 1's copy (code 7db302b, its own scripts), so the
#  checkout's state does not matter; --mem=128G overrides the copy's 64 GB.
#
#    bash resume_run1_exp3.sh check    read-only preflight
#    bash resume_run1_exp3.sh submit   submit the model-loop job
# ============================================================================
set -euo pipefail
RUN_COMMIT=7db302b60151e4997d851b969ba8a00f5b230da4
RUN=run1
EXPECTED_SITE=auto-psych-2c5da-0926-run1
EXPECTED_STUDY3=<study id>
OWNER=kushin
MEM=128G
TIME=1-12:00:00   # one model loop on 7,680 trials; normal QOS (long needs >= 48 h)

MODE="${1:-}"
case "$MODE" in check|submit) ;; *) echo "usage: bash $0 check|submit" >&2; exit 2 ;; esac
die() { echo "STOP: $*" >&2; exit 1; }

source "$HOME/repos/live_env.sh"          # REPO (for .secrets), OUTER_LIVE_SLURM_DIR
source "$OUTER_LIVE_SLURM_DIR/_env.sh"    # WORK_ROOT, VENV_PY, modules, secrets
source "$OUTER_LIVE_SLURM_DIR/_hosting_site.sh"
WT="$WORK_ROOT/runs/$RUN/repo"
OUT="$WORK_ROOT/$RUN/data"
E="$OUT/subjective_randomness"
LOGDIR="$WORK_ROOT/slurm_logs"
py() { "$VENV_PY" -c "$@"; }

echo "--- jobs"
[[ -z "$(squeue --me -h -o '%j' | grep -x "outer_live_${RUN}.*" || true)" ]] || die "an outer_live_${RUN} job is queued or running"
echo "  no run-1 job queued or running"

echo "--- run copy"
prov="$(py 'import json,sys; p=json.load(open(sys.argv[1])); print(p["git_commit"], p["git_dirty"])' "$WT/code_provenance.json")"
[[ "$prov" == "$RUN_COMMIT False" ]] || die "run copy provenance is '$prov', expected '$RUN_COMMIT False'"
grep -q 'def _new_participant_draws' "$WT/src/pipelines/outer_loop/eig.py" || die "run copy lacks the design fix"
echo "  $WT at $RUN_COMMIT"

echo "--- run 1's tree"
rows=$(( $(wc -l < "$E/experiment3/data/responses.csv") - 1 ))
[[ "$rows" -eq 2560 ]] || die "experiment 3 has $rows rows"
study="$(py 'import json,sys; print(json.load(open(sys.argv[1]))["prolific_study_id"])' "$E/experiment3/deployment/deployment_manifest.json")"
[[ "$study" == "$EXPECTED_STUDY3" ]] || die "experiment 3 records study $study"
[[ -d "$E/experiment3/cognitive_models_input" ]] || die "experiment 3 has no recorded model-loop input"
[[ ! -f "$E/experiment3/model_loop/export_complete.json" ]] || die "experiment 3's model loop already finished"
(cd "$WT" && py 'import sys, pathlib; from src.pipelines.outer_loop.orchestrator_validators import validate_cc_output as v; ok, m = v("5_model_loop", pathlib.Path(sys.argv[1])); sys.exit(0 if ok else m)' "$E/experiment2") \
  || die "experiment 2's model loop does not validate"
echo "  experiment 3: $rows rows, study $study, recorded input present, loop unfinished; experiment 2 validates"

echo "--- config"
cfg_env="$("$VENV_PY" "$OUTER_LIVE_SLURM_DIR/_pilot_config.py" "$REPO/full_run_sr-oct26.yaml" --check 2>/dev/null)" \
  || die "config / Prolific token validation failed"
eval "$cfg_env"
[[ "$N_PARTICIPANTS" == 40 && "$PROLIFIC_MODE" == live && "$CODING_AGENT" == claude && "$CLAUDE_AUTH" == api \
   && "$CODING_AGENT_MODEL" == claude-opus-5-5 && "$INNER_LOOP_ITERATIONS" == 5 && "$INNER_LOOP_CANDIDATES" == 6 \
   && "$PRUNE_DSE_MULTIPLIER" == 4 ]] || die "config settings differ from run 1's"
SITE="$(hosting_site "$FIREBASE_PROJECT" "$RUN_LABEL" "$RUN")"
[[ "$SITE" == "$EXPECTED_SITE" ]] || die "site $SITE"
unset EXPERIMENTS
echo "  N=$N_PARTICIPANTS agent=$CODING_AGENT_MODEL rounds=$INNER_LOOP_ITERATIONS x $INNER_LOOP_CANDIDATES prune=${PRUNE_DSE_MULTIPLIER}·dse"

EXPORT="ALL,RUN_LABEL=$RUN,RUN_WORKTREE=$WT,CODING_AGENT=$CODING_AGENT,AUTO_PSYCH_OUTPUT_DIR=$OUT,AUTO_PSYCH_HOSTING_SITE=$SITE,AUTO_PSYCH_COLLECTION_OWNER=$OWNER,EXPERIMENT=3,RESUME_AGENTS=5_model_loop,OUTER_LIVE_SLURM_DIR=$WT/scripts/outer_loop_live"
ARGS=(--job-name="outer_live_${RUN}_C" --output="$LOGDIR/%x_%j.out" --error="$LOGDIR/%x_%j.out"
      --mem="$MEM" --time="$TIME")
case "$MODE" in
  check)
    sbatch --test-only "${ARGS[@]}" "$WT/scripts/outer_loop_live/run_live.sbatch" 2>&1 | tail -1
    echo "  would submit: sbatch ${ARGS[*]} --export=$EXPORT $WT/scripts/outer_loop_live/run_live.sbatch"
    ;;
  submit)
    jc=$(sbatch --parsable "${ARGS[@]}" --export="$EXPORT" "$WT/scripts/outer_loop_live/run_live.sbatch")
    echo "submitted experiment 3's model loop: job $jc (log $LOGDIR/outer_live_${RUN}_C_$jc.out)"
    ;;
esac
