#!/bin/bash
# Archive a finished cell's agent _runs, then remove its agent tree -- only
# once the archive is written and reads back.
#
#   archive_agent_tree.sh <task label> <run_dir> <agent_dir>
#
# The agent tree (<agent_dir>/repo: sanitized source + the agents' _runs) is
# thousands of small files, the biggest inode consumer per task. Its _runs go
# to <run_dir>/agent_runs.tar.gz and the tree is deleted. The archive is
# written under a temporary name, listed back with `tar tzf`, and only then
# moved into place; the tree is removed only after that. A failed tar
# (quota, inodes) used to be followed by the `rm -rf` all the same, deleting
# the only copy of the agents' work. Now the tree stays, and this says so.
# An existing archive is never overwritten (it may be the only record of an
# earlier run), and the tree is then kept too. Exits 0 either way: the cell's
# results are already written, and a kept tree only costs inodes.
set -uo pipefail
[[ $# -eq 3 ]] || { echo "usage: $0 <task label> <run_dir> <agent_dir>" >&2; exit 2; }
task="$1"; run_dir="$2"; agent_dir="$3"
archive="$run_dir/agent_runs.tar.gz"
partial="$archive.partial"

keep_tree() {
  local why="$1"
  {
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "[task $task] ERROR: $why"
    echo "[task $task] KEEPING the agent tree $agent_dir (it is the only copy of"
    echo "[task $task] the agents' _runs). Archive it by hand and delete it once"
    echo "[task $task] the archive reads back: tar tzf $archive"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  } | tee /dev/stderr
  exit 0
}

if [[ -e "$archive" ]]; then
  echo "[task $task] WARNING: $archive already exists; not overwriting it, and keeping the agent tree $agent_dir"
  exit 0
fi

if [[ -d "$agent_dir/repo/_runs" ]]; then
  rm -f "$partial"
  if ! tar czf "$partial" -C "$agent_dir/repo" _runs; then
    rm -f "$partial"
    keep_tree "tar failed writing $archive"
  fi
  if ! tar tzf "$partial" 2>/dev/null | grep -qx '_runs/'; then
    rm -f "$partial"
    keep_tree "$archive was written but does not read back (tar tzf)"
  fi
  mv "$partial" "$archive" || keep_tree "could not move the archive into place at $archive"
  echo "[task $task] archived agent _runs -> $archive"
fi

# rm -rf unlinks the venv and mcmc_cache symlinks inside the agent dir without
# following them: the shared venv and this cell's cache are untouched.
rm -rf "$agent_dir"
rm -f "$run_dir/repo"
echo "[task $task] removed the agent tree to reclaim inodes: $agent_dir"
