#!/usr/bin/env bash
# Run an arbitrary shell command on the remote cluster, inside the project directory.
# Adapted from standard_model_2/sherlock/remote_exec.sh.
#
# Usage:  ./remote_jobs/remote_exec.sh "<remote shell command>"
# Examples:
#   ./remote_jobs/remote_exec.sh "git pull && squeue -u \$USER"
#   ./remote_jobs/remote_exec.sh "ls -la \$SCRATCH/auto-psych/projects"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

if [ -f "$SCRIPT_DIR/.env.local" ]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.env.local"
fi

REMOTE_HOST="${REMOTE_HOST:-sherlock}"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-\$HOME/auto-psych}"

CMD="${*:-}"
if [ -z "$CMD" ]; then
  echo "Usage: $0 \"<remote shell command>\"" >&2
  exit 1
fi

# Quote the entire remote command so that $HOME / $SCRATCH / etc. expand on the server.
ssh "$REMOTE_HOST" "cd ${REMOTE_PROJECT_DIR} && ${CMD}"
