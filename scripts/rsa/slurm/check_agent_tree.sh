#!/bin/bash
# Verify an RSA agent tree before any agent starts. Fails (exit 1) on the
# first class of problem found, listing every offending path.
#
#   check_agent_tree.sh <tree> [--gt-file <pristine seed .py>]... [--forbid-file <file>]...
#
# --gt-file      a seed the cell withholds (the recovery ground truth, its
#                near-twins). Its name appears in generic text (the report
#                template, the primer), so the check is on code, not names:
#                no file named <name>.py, no manifest entry `name: <name>`
#                (outside _runs/), and no byte-identical copy anywhere,
#                _runs/ included.
# --forbid-file  a file agents must never read (the cell's test.csv, the
#                simulation's provenance.json, split.json): no byte-identical
#                copy and no file of the same name anywhere in the tree.
#
# Also checked: the paths agent_tree.exclude must have removed are absent,
# no CSV is left outside _runs/ (all data comes from the loop's results
# dir), and the tree has what the agents need (.here, opencode.json with
# snapshots off, the self-check module, the uv shim).
set -uo pipefail
[[ $# -ge 1 ]] || { echo "usage: $0 <tree> [--gt-file F]... [--forbid-file F]..." >&2; exit 2; }
tree="${1%/}"; shift
gt_files=(); forbid_files=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gt-file) gt_files+=("$2"); shift 2 ;;
    --forbid-file) forbid_files+=("$2"); shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -d "$tree" ]] || { echo "ERROR: $tree is not a directory" >&2; exit 1; }

fail() {
  echo "ERROR (agent tree $tree): $1" >&2
  shift
  [[ $# -gt 0 ]] && printf '  %s\n' "$@" >&2
  exit 1
}

# Files whose bytes equal <file>, under <tree> (size first, then cmp).
identical_copies() {
  local file="$1" size
  size=$(wc -c < "$file" | tr -d ' ')
  find "$tree" -type f -size "${size}c" -print0 2>/dev/null \
    | while IFS= read -r -d '' candidate; do
        cmp -s "$file" "$candidate" && echo "${candidate#$tree/}"
      done
}

# 1. What the agents need.
[[ -f "$tree/.here" ]] || fail "no .here (pyprojroot cannot find the tree's root)"
[[ -f "$tree/src/rsa/loop/check_candidate.py" ]] || fail "no src/rsa/loop/check_candidate.py (the agents' self-check)"
[[ -f "$tree/opencode.json" ]] || fail "no opencode.json"
grep -Eq '"snapshot"[[:space:]]*:[[:space:]]*false' "$tree/opencode.json" \
  || fail "opencode.json does not set \"snapshot\": false (each agent would keep a git snapshot of the tree)"
[[ -x "$tree/.agent_bin/uv" ]] || fail "no .agent_bin/uv (scrub_agent_tree.sh didn't run)"

# 2. What agent_tree.exclude must have removed.
present=()
for rel in data docs tests scripts .git .secrets CLAUDE.md AGENTS.md \
           src/pipelines/outer_loop/projects/rsa_reference/data \
           src/pipelines/outer_loop/projects/rsa_reference/experiment \
           src/rsa/ingest src/rsa/split.py src/rsa/simulate.py \
           src/rsa/recovery.py src/rsa/evaluate_heldout.py src/rsa/pragmods_ingest.py; do
  [[ -e "$tree/$rel" ]] && present+=("$rel")
done
[[ ${#present[@]} -eq 0 ]] || fail "paths agent_tree.exclude should have removed are present:" "${present[@]}"
mapfile -t secrets < <(find "$tree" \( -name '.secrets*' -o -name '*.secrets' \) -print | sed "s|^$tree/||")
[[ ${#secrets[@]} -eq 0 ]] || fail "secrets files (or backups of one) in the tree:" "${secrets[@]}"
mapfile -t csvs < <(find "$tree" -path "$tree/_runs" -prune -o -type f \( -name '*.csv' -o -name '*.csv.gz' \) -print | sed "s|^$tree/||")
[[ ${#csvs[@]} -eq 0 ]] || fail "CSV files outside _runs/ (agents must see only their cell's training data):" "${csvs[@]}"

# 3. The withheld seeds' code (none in a real cell: the guarded expansions
#    below, as el7's bash 4.2 treats an empty array as unbound under set -u).
for gt_file in ${gt_files[@]+"${gt_files[@]}"}; do
  [[ -f "$gt_file" ]] || fail "--gt-file $gt_file does not exist (the pristine copy is needed to check for copies)"
  name="$(basename "$gt_file" .py)"
  mapfile -t named < <(find "$tree" -path "$tree/_runs" -prune -o -type f -name "$name.py" -print | sed "s|^$tree/||")
  [[ ${#named[@]} -eq 0 ]] || fail "files named $name.py (a withheld seed):" "${named[@]}"
  mapfile -t listed < <(find "$tree" -path "$tree/_runs" -prune -o -type f -name models_manifest.yaml -print0 \
                          | xargs -0 -r grep -lE "^[[:space:]]*(-[[:space:]]*)?name:[[:space:]]*['\"]?$name['\"]?[[:space:]]*$" \
                          | sed "s|^$tree/||")
  [[ ${#listed[@]} -eq 0 ]] || fail "manifests listing the withheld seed $name:" "${listed[@]}"
  mapfile -t copies < <(identical_copies "$gt_file")
  [[ ${#copies[@]} -eq 0 ]] || fail "byte-identical copies of the withheld seed $name:" "${copies[@]}"
  echo "[check] withheld seed $name: no file, no manifest entry, no copy"
done

# 4. Files agents must never read.
for forbidden in ${forbid_files[@]+"${forbid_files[@]}"}; do
  [[ -f "$forbidden" ]] || fail "--forbid-file $forbidden does not exist"
  base="$(basename "$forbidden")"
  mapfile -t same_name < <(find "$tree" -type f -name "$base" -print | sed "s|^$tree/||")
  [[ ${#same_name[@]} -eq 0 ]] || fail "files named $base in the tree:" "${same_name[@]}"
  mapfile -t copies < <(identical_copies "$forbidden")
  [[ ${#copies[@]} -eq 0 ]] || fail "byte-identical copies of $forbidden:" "${copies[@]}"
  echo "[check] $base: absent from the tree"
done
echo "[check] agent tree OK: $tree"
