#!/bin/bash
#SBATCH --job-name=rsa_archive
#SBATCH --partition=mcfrank
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#
# Archive a finished RSA sweep off $SCRATCH (purged after 90 days) for the
# paper: everything the repository does not hold. From the checkout:
#
#   mkdir -p "$WORK_ROOT/logs"
#   sbatch --chdir="$HOME/auto-psych" -o "$WORK_ROOT/logs/%x_%j.out" scripts/rsa/slurm/archive_sweep.sbatch
#
# Writes to $ARCHIVE_ROOT/<sweep> (default $OAK/auto-psych/archive when Oak is
# mounted, else $GROUP_HOME/auto-psych/archive):
#   data.tar.gz            $DATA_ROOT: combined trials, splits, recovery simulations and
#                          their provenance (trial-level: private, never public)
#   records.tar.gz         $CELLS_ROOT (cell records, outputs, attempts) and $WORK_ROOT/logs
#   cells/<cell>.tar.gz    the loop directory: every round's candidates with the agents'
#                          logs and opencode sessions, ledger, history, CV folds, steps
#   fits/<cell>.tar.gz     the loop's .fit_cache (re-scoring without refitting); ARCHIVE_FITS=0 skips
#   SHA256SUMS, MANIFEST.json   checksums; sizes, file counts, code commit, date
# It also rewrites each cell's agent_activity.md with web tool calls listed
# apart from URLs in log text (into the cell record and outputs/, so the
# bring-back picks it up). Before anything is written it searches every file
# it will archive for the agents' API key and stops if one holds it.
# Idempotent: an existing archive is refused unless OVERWRITE=1.
set -euo pipefail
export REPO="${REPO:-${SLURM_SUBMIT_DIR:-$HOME/auto-psych}}"
source "$REPO/scripts/rsa/slurm/_env.sh"
SWEEP="$(basename "$WORK_ROOT")"
DEFAULT_ARCHIVE="${GROUP_HOME:?no GROUP_HOME}/auto-psych/archive"
[[ -n "${OAK:-}" && -d "${OAK:-}" ]] && DEFAULT_ARCHIVE="$OAK/auto-psych/archive"
DEST="${ARCHIVE_ROOT:-$DEFAULT_ARCHIVE}/$SWEEP"
ARCHIVE_FITS="${ARCHIVE_FITS:-1}"
if [[ -e "$DEST/MANIFEST.json" && "${OVERWRITE:-0}" != 1 ]]; then
  echo "ERROR: $DEST is already archived (OVERWRITE=1 to redo)" >&2; exit 1
fi
mkdir -p "$DEST/cells" "$DEST/fits"
echo "[archive] $WORK_ROOT -> $DEST"

# Every cell's loop directory, in the agent tree.
declare -A LOOP
for (( task = 0; task < RSA_N_CELLS; task++ )); do
  rsa_cell "$task"
  [[ -f "$CELLS_ROOT/$CELL/DONE" ]] || { echo "ERROR: $CELL is not finished" >&2; exit 1; }
  LOOP[$CELL]="$AGENT_TREES_ROOT/$(cat "$CELLS_ROOT/$CELL/agent_tree_id")/repo/_runs/loop"
  [[ -d "${LOOP[$CELL]}" ]] || { echo "ERROR: ${LOOP[$CELL]} is gone" >&2; exit 1; }
done

# 1. No secrets: the agents' key must appear in no file to be archived.
KEY="${GOOGLE_GENERATIVE_AI_API_KEY:-${GOOGLE_API_KEY:-}}"
[[ -n "$KEY" ]] || { echo "ERROR: no API key in the environment to scan for" >&2; exit 1; }
hits=$(grep -rlF --exclude-dir=.home -- "$KEY" "$DATA_ROOT" "$CELLS_ROOT" "$WORK_ROOT/logs" "${LOOP[@]}" 2>/dev/null || true)
if [[ -n "$hits" ]]; then
  echo "ERROR: these files hold the agents' API key; nothing was archived:" >&2
  echo "$hits" >&2
  exit 1
fi
echo "[archive] key scan: clean"

# 2. agent_activity.md with web tool calls apart from URLs in log text.
for cell in "${!LOOP[@]}"; do
  agent_dir="$(dirname "$(dirname "$(dirname "${LOOP[$cell]}")")")"
  "$VENV_PY" "$REPO/scripts/subjective_randomness/slurm/agent_activity_report.py" --agent-dir "$agent_dir" \
    --out "$CELLS_ROOT/$cell/agent_activity.md" --expected "$HOME" "$(dirname "$(dirname "$VENV_PY")")"
  cp "$CELLS_ROOT/$cell/agent_activity.md" "$CELLS_ROOT/$cell/outputs/agent_activity.md"
done

# 3. The archives.
tar -czf "$DEST/data.tar.gz" -C "$(dirname "$DATA_ROOT")" "$(basename "$DATA_ROOT")"
tar -czf "$DEST/records.tar.gz" --exclude='repo' --exclude='results' -C "$WORK_ROOT" "$(basename "$CELLS_ROOT")" logs code_commit
for cell in "${!LOOP[@]}"; do
  tar -czf "$DEST/cells/$cell.tar.gz" --exclude='.fit_cache' --exclude='.home' --exclude='.agent_shell' \
    -C "$(dirname "${LOOP[$cell]}")" loop --transform "s,^loop,$cell,"
  if [[ "$ARCHIVE_FITS" == 1 ]]; then
    tar -czf "$DEST/fits/$cell.tar.gz" -C "${LOOP[$cell]}" .fit_cache
  fi
done

# 4. Checksums and a manifest.
(cd "$DEST" && find . -name '*.tar.gz' -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
"$VENV_PY" - "$DEST" "$WORK_ROOT" <<'PY'
import json, os, sys, tarfile, datetime, pathlib
dest, work = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
files = {}
for p in sorted(dest.rglob("*.tar.gz")):
    with tarfile.open(p) as t:
        members = [m for m in t.getmembers() if m.isfile()]
    files[str(p.relative_to(dest))] = dict(bytes=p.stat().st_size, files=len(members),
                                           unpacked_bytes=sum(m.size for m in members))
json.dump(dict(sweep=work.name, work_root=str(work), code_commit=(work / "code_commit").read_text().strip(),
               archived=datetime.datetime.now().isoformat(timespec="seconds"), archives=files),
          open(dest / "MANIFEST.json", "w"), indent=1)
PY
du -sh "$DEST"
echo "done: $DEST (record SHA256SUMS and MANIFEST.json in the repository's sweep folder)"
