#!/bin/bash
# Look for the held-out ground truth's name in an agent tree or in agent output.
#
#   scan_gt_name.sh --before-agents <gt> <agent_tree> <mentions_file>
#   scan_gt_name.sh --after-run     <gt> <agent_runs_dir> <mentions_file>
#
# --before-agents: the tree agents are about to receive (their _runs/ results
#   root excluded). A name here is a leak on our side — the exclude list missed
#   a file — so the cell stops before any agent has run.
# --after-run: what the agents wrote. A name here may be a coincidence (an
#   agent can coin "motif_stack" unaided), so the scan only lists the files in
#   <mentions_file> and warns; it never fails the cell.
set -uo pipefail
[[ $# -eq 4 ]] || { echo "usage: $0 --before-agents|--after-run <gt> <dir> <mentions_file>" >&2; exit 2; }
mode="$1"; gt="$2"; dir="$3"; mentions="$4"

case "$mode" in
  --before-agents)
    hits=$(grep -rIl --exclude-dir=_runs -- "$gt" "$dir" | sed "s|^$dir/||")
    if [[ -n "$hits" ]]; then
      echo "ERROR: the agent tree names the held-out ground truth '$gt' (add these to agent_tree.exclude):" >&2
      printf '  %s\n' $hits >&2
      exit 1
    fi
    ;;
  --after-run)
    hits=$(grep -rIl -- "$gt" "$dir")
    if [[ -n "$hits" ]]; then
      printf '%s\n' $hits > "$mentions"
      echo "WARNING: agent output names the held-out ground truth '$gt' in $(wc -l < "$mentions") file(s); list: $mentions"
    else
      rm -f "$mentions"
    fi
    ;;
  *)
    echo "unknown mode: $mode" >&2; exit 2 ;;
esac
