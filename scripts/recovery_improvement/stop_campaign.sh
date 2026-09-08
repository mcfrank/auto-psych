#!/bin/bash
# Stop a recovery-improvement campaign: no further review job will run.
#
#   bash scripts/recovery_improvement/stop_campaign.sh <campaign_name|campaign_root> [--cancel-sweeps]
#
# Writes STOP into the campaign root (a queued review job that starts later
# exits at once), cancels pending review jobs, and lists the sweep jobs still
# running. Sweeps are left running unless --cancel-sweeps is given — a
# finished sweep is still worth reading.
set -euo pipefail
[[ $# -ge 1 ]] || { echo "usage: $0 <campaign_name|campaign_root> [--cancel-sweeps]" >&2; exit 1; }
target="$1"; cancel_sweeps="${2:-}"
if [[ -d "$target" ]]; then CAMPAIGN_ROOT="$target"; else CAMPAIGN_ROOT="${SCRATCH:-$GROUP_SCRATCH}/auto-psych/recovery_improvement/$target"; fi
[[ -f "$CAMPAIGN_ROOT/campaign.env" ]] || { echo "ERROR: no campaign at $CAMPAIGN_ROOT" >&2; exit 1; }
set -a; source "$CAMPAIGN_ROOT/campaign.env"; set +a

echo "stopped by $USER on $(date '+%Y-%m-%d %H:%M:%S')" > "$CAMPAIGN_ROOT/STOP"
echo "wrote $CAMPAIGN_ROOT/STOP"

pending=$(squeue --me -h -o "%i %j %t" | awk -v n="recovery_review_$CAMPAIGN_NAME" '$2==n && $3!="R"{print $1}')
if [[ -n "$pending" ]]; then
  echo "cancelling queued review jobs: $pending"
  scancel $pending
fi
running=$(squeue --me -h -o "%i %j %t" | awk -v n="recovery_review_$CAMPAIGN_NAME" '$2==n && $3=="R"{print $1}')
[[ -n "$running" ]] && echo "NOTE: review job(s) still running (not cancelled; they exit without launching once done): $running"

sweep_ids=""
for jobs in "$CAMPAIGN_ROOT"/iter*/jobs.json; do
  [[ -f "$jobs" ]] || continue
  ids=$(jq -r '.sweep_jobs // {} | [.setup_id, .array_id, .analysis_id] | map(select(. != null)) | join(" ")' "$jobs")
  sweep_ids="$sweep_ids $ids"
done
active=""
for id in $sweep_ids; do
  squeue -h -j "$id" -o "%i" 2>/dev/null | grep -q . && active="$active $id"
done
if [[ -n "$active" ]]; then
  if [[ "$cancel_sweeps" == "--cancel-sweeps" ]]; then
    echo "cancelling sweep jobs:$active"; scancel $active
  else
    echo "sweep jobs still active (left running; pass --cancel-sweeps to kill them):$active"
  fi
fi
echo "done. Journal: $CAMPAIGN_ROOT/journal.md"
