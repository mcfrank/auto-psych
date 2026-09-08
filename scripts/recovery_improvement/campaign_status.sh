#!/bin/bash
# Show where a recovery-improvement campaign stands (login-node safe: bash + jq).
#
#   bash scripts/recovery_improvement/campaign_status.sh <campaign_name|campaign_root>
set -euo pipefail
[[ $# -ge 1 ]] || { echo "usage: $0 <campaign_name|campaign_root>" >&2; exit 1; }
target="$1"
if [[ -d "$target" ]]; then CAMPAIGN_ROOT="$target"; else CAMPAIGN_ROOT="${SCRATCH:-$GROUP_SCRATCH}/auto-psych/recovery_improvement/$target"; fi
[[ -f "$CAMPAIGN_ROOT/campaign.env" ]] || { echo "ERROR: no campaign at $CAMPAIGN_ROOT" >&2; exit 1; }
set -a; source "$CAMPAIGN_ROOT/campaign.env"; set +a

echo "campaign $CAMPAIGN_NAME  ($CAMPAIGN_ROOT)"
echo "  base $BASE_BRANCH @ ${BASE_COMMIT:0:10}   max iterations $MAX_ITERATIONS   model $REVIEW_MODEL"
[[ -f "$CAMPAIGN_ROOT/STOP" ]] && echo "  STOPPED: $(cat "$CAMPAIGN_ROOT/STOP")"
echo

summarise_sweep() {  # <test_retest.json>
  jq -r '"    ICC(2,1) \(.icc_2_1 // "n/a" | if type=="number" then (.*1000|round/1000) else . end)   " +
         ([.gt_models[] as $g | "\($g)=\(.per_gt_model[$g].mean*1000|round/1000)"] | join("  "))' "$1"
}

echo "baseline (reference): ${BASELINE_ROOTS%% *}"
[[ -f "${BASELINE_ROOTS%% *}/test_retest.json" ]] && summarise_sweep "${BASELINE_ROOTS%% *}/test_retest.json"
echo

for iter_dir in $(ls -d "$CAMPAIGN_ROOT"/iter* 2>/dev/null | sort -V); do
  n="${iter_dir##*/iter}"
  echo "iteration $n"
  [[ -f "$iter_dir/prescription.md" ]] && echo "  prescription: $iter_dir/prescription.md" || echo "  prescription: (none yet)"
  if [[ -f "$iter_dir/STOP" ]]; then echo "  decision: STOP — $(head -c 200 "$iter_dir/STOP")"; fi
  if [[ -f "$iter_dir/next_run.env" ]]; then
    echo "  decision: sweep  ($(grep -v '^#' "$iter_dir/next_run.env" | grep . | tr '\n' ' ' || true))"
  fi
  if [[ -f "$iter_dir/jobs.json" ]]; then
    ids=$(jq -r '[(.sweep_jobs // {} | .setup_id, .array_id, .analysis_id), .next_review_job] | map(select(. != null)) | join(",")' "$iter_dir/jobs.json")
    if [[ -n "$ids" ]]; then
      echo "  jobs (id name state elapsed):"
      sacct -X -n -j "$ids" --format=JobID%14,JobName%28,State%16,Elapsed 2>/dev/null | sed 's/^/    /' | sort -u
    fi
  fi
  if [[ -f "$iter_dir/sweep/test_retest.json" ]]; then
    echo "  sweep result:"; summarise_sweep "$iter_dir/sweep/test_retest.json"
  elif [[ -d "$iter_dir/sweep" ]]; then
    done_n=$(ls "$iter_dir"/sweep/run*/*/holdout.csv 2>/dev/null | wc -l)
    echo "  sweep result: not aggregated yet ($done_n cells finished)"
  fi
  echo
done
[[ -f "$CAMPAIGN_ROOT/final_digest.md" ]] && echo "final digest: $CAMPAIGN_ROOT/final_digest.md"
echo "journal: $CAMPAIGN_ROOT/journal.md"
