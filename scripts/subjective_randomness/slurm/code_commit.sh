#!/bin/bash
# Print the identity of the code in a checkout: its commit, plus a hash of the
# uncommitted changes when there are any.
#
#   code_commit.sh <repo>     e.g. 3f2c...  or  3f2c...-dirty-1a2b3c4d5e6f
#
# The setup job records this for the code it stages ($WORK_ROOT/code_commit),
# and every cell records the code it started on (run<r>/<gt>/code_commit): a
# cell is never resumed on other code. (`cd`, not `git -C`: el7's git 1.8
# predates -C.)
set -euo pipefail
[[ $# -eq 1 ]] || { echo "usage: $0 <repo>" >&2; exit 2; }
cd "$1"
commit=$(git rev-parse HEAD)
changes=$(git status --porcelain; git diff HEAD)
if [[ -n "$changes" ]]; then
  echo "$commit-dirty-$(printf '%s' "$changes" | sha256sum | cut -c1-12)"
else
  echo "$commit"
fi
