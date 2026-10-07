#!/bin/bash
# Finish an RSA agent tree after its rsync copy (build_agent_tree.sh): remove
# the cell's excluded seed models (the recovery ground truth and its
# near-twins) with their manifest entries, and give the tree a working
# `uv run python ...` for the agents' self-check.
#
#   scrub_agent_tree.sh <tree> <agent_python> [excluded_seed ...]
#
# <agent_python> is the venv interpreter the harness runs with, by the path
# the sandbox mounts it at (the cell's opaque <agent_dir>/venv/bin/python).
#
# Why a uv shim: the agents' brief and primer give the self-check as
# `uv run python -m src.rsa.loop.check_candidate <dir> --responses <csv>`,
# run from the tree. Inside the sandbox the tree and the venv are read-only,
# so the real uv, which creates or syncs <tree>/.venv against uv.lock, cannot
# run. (The subjective-randomness brief prints the harness's interpreter
# instead, so its sweeps never needed this.) The shim serves `uv run python`
# (and `uv run <file>.py`) with the harness's environment and refuses every
# other uv command loudly; the job puts <tree>/.agent_bin first on PATH. The
# tree's .venv also links to that environment, for an agent that looks there.
set -euo pipefail
[[ $# -ge 2 ]] || { echo "usage: $0 <tree> <agent_python> [excluded_seed ...]" >&2; exit 2; }
tree="$1"; agent_python="$2"; shift 2
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/_cells.sh"
remove_entry="$here/../../subjective_randomness/remove_manifest_entry.py"
[[ -f "$remove_entry" ]] || { echo "ERROR: $remove_entry not found" >&2; exit 1; }
[[ -x "$agent_python" ]] || { echo "ERROR: $agent_python is not an executable interpreter" >&2; exit 1; }
seeds_dir="$tree/$RSA_SEED_MODELS_REL"
[[ -f "$seeds_dir/models_manifest.yaml" ]] || { echo "ERROR: no seed manifest at $seeds_dir" >&2; exit 1; }

# pyprojroot.here() sentinel: .git is not copied.
touch "$tree/.here"

for seed in "$@"; do
  rm -f "$seeds_dir/$seed.py"
  [[ ! -e "$seeds_dir/$seed.py" ]] || { echo "ERROR: could not remove $seeds_dir/$seed.py" >&2; exit 1; }
  # Its manifest line is the model's name and a sentence stating its
  # mechanism. Run from the tree so pyprojroot resolves the tree's src/.
  # (No --missing-ok: every excluded seed must be listed.)
  (cd "$tree" && "$agent_python" "$remove_entry" --models-dir "$seeds_dir" --name "$seed")
  echo "[scrub] removed seed $seed (file and manifest entry) from the agent tree"
done

venv_dir="$(dirname "$(dirname "$agent_python")")"
ln -sfn "$venv_dir" "$tree/.venv"
mkdir -p "$tree/.agent_bin"
ln -sfn "$agent_python" "$tree/.agent_bin/python"
ln -sfn "$agent_python" "$tree/.agent_bin/python3"
cat > "$tree/.agent_bin/uv" <<SHIM
#!/bin/bash
# uv for the agents' sandbox. The tree and its Python environment are
# read-only here, so nothing can be installed or synced: \`uv run python ...\`
# runs the environment's interpreter, and every other uv command is refused.
PY='$agent_python'
SHIM
cat >> "$tree/.agent_bin/uv" <<'SHIM'
refuse() {
  echo "uv (sandbox): '$*' is not available here. The environment is fixed and read-only;" >&2
  echo "use 'uv run python ...' (e.g. uv run python -m src.rsa.loop.check_candidate <dir> --responses <csv>)." >&2
  exit 2
}
case "${1:-}" in
  --version|-V|version) echo "uv (sandbox shim; runs $PY)"; exit 0 ;;
  run) shift ;;
  *) refuse "$@" ;;
esac
while [[ $# -gt 0 ]]; do
  case "$1" in
    --frozen|--locked|--no-sync|--offline|--quiet|-q|--no-project|--active|--isolated) shift ;;
    --) shift; break ;;
    -*) refuse "run $1" ;;
    *) break ;;
  esac
done
case "${1:-}" in
  python|python3|python3.*) shift; exec "$PY" "$@" ;;
  *.py) exec "$PY" "$@" ;;
  "") refuse "run" ;;
  *) refuse "run $1" ;;
esac
SHIM
chmod +x "$tree/.agent_bin/uv"
echo "[scrub] agent tools: $tree/.agent_bin/{uv,python,python3} -> $agent_python"
