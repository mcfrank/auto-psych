#!/bin/bash
# Build (or refresh) one cell's agent tree: an rsync copy of agent_src with
# agent_tree.exclude applied, then scrub_agent_tree.sh. The caller runs
# check_agent_tree.sh on the result before any agent starts.
#
#   build_agent_tree.sh <agent_src> <tree> <agent_python> [excluded_seed ...]
#
# --delete-excluded also cleans a tree built before an exclusion was added;
# the protect filter keeps the loop's results under <tree>/_runs (whose
# .fit_cache/*.nc match an exclude pattern).
set -euo pipefail
[[ $# -ge 3 ]] || { echo "usage: $0 <agent_src> <tree> <agent_python> [excluded_seed ...]" >&2; exit 2; }
agent_src="$1"; tree="$2"; agent_python="$3"; shift 3
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exclude="$here/agent_tree.exclude"
[[ -f "$exclude" ]] || { echo "ERROR: $exclude not found" >&2; exit 1; }
[[ -d "$agent_src/src" ]] || { echo "ERROR: $agent_src has no src/ (stage_code.sh didn't run?)" >&2; exit 1; }
command -v rsync >/dev/null || { echo "ERROR: rsync is not on PATH" >&2; exit 1; }
mkdir -p "$tree"
rsync -a --delete --delete-excluded --filter='P /_runs/***' \
  --exclude-from="$exclude" "$agent_src"/ "$tree"/
bash "$here/scrub_agent_tree.sh" "$tree" "$agent_python" "$@"
