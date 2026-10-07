#!/bin/bash
# Per-cell state of the RSA run-1 sweep (login node; reads files only).
#
#   bash scripts/rsa/slurm/rsa_status.sh [WORK_ROOT]
#
# For each cell: state (pending / running / done / failed), the stage the
# loop is in, the current best model, the ledger's admitted / rejected /
# pruned counts (attempted_hypotheses.jsonl), the agents' token spend
# (token_usage.jsonl) and, when scored, the recovery verdict.
#   running  the cell's lock is held by a job squeue still lists
#   done     DONE written (loop, held-out and recovery scoring all finished)
#   failed   the last job exited non-zero (see last_exit and its log)
#   stopped  the last job exited 0 without finishing (e.g. a duplicate)
#   pending  no job has started it
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/_cells.sh"
WORK_ROOT="${1:-${WORK_ROOT:-${SCRATCH:-${GROUP_SCRATCH:?set WORK_ROOT (or SCRATCH)}}/auto-psych/rsa_run1}}"
[[ -d "$WORK_ROOT" ]] || { echo "no sweep at $WORK_ROOT" >&2; exit 1; }
PY="$WORK_ROOT/venv/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python3)"

# Job ids Slurm still knows (pending or running), for the cells' locks.
live_jobs=""
if command -v squeue >/dev/null; then
  live_jobs="$(squeue -h -u "$USER" -o '%A %i' 2>/dev/null | tr ' ' '\n' | sort -u | tr '\n' ' ')"
fi

cells=()
for (( t = 0; t < RSA_N_CELLS; t++ )); do rsa_cell "$t"; cells+=("$t:$CELL"); done
echo "sweep: $WORK_ROOT   code: $(cat "$WORK_ROOT/code_commit" 2>/dev/null || echo 'not staged')"
"$PY" - "$WORK_ROOT" "$live_jobs" "${cells[@]}" <<'PY'
import json
import sys
from pathlib import Path

root, live = Path(sys.argv[1]), set(sys.argv[2].split())
rows = []


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


for item in sys.argv[3:]:
    task, cell = item.split(":", 1)
    d = root / "cells" / cell
    res = d / "results"
    lock = d / ".cell_lock"
    last = (d / "last_exit").read_text().split() if (d / "last_exit").exists() else []
    if (d / "DONE").exists():
        state = "done"
    elif lock.exists() and lock.read_text().strip() in live:
        state = "running"
    elif last:
        state = "failed" if last[0] != "0" else "stopped"
    else:
        state = "pending"
    cfg = read_json(d / "cell.json") or {}
    history = read_json(res / "history.json") or []
    n_rounds = cfg.get("max_iterations", 5)
    rounds_done = sum(1 for h in history if 0 <= h.get("round", -1) < n_rounds)
    if (res / "recovery" / "recovery.json").exists() or (res / "heldout" / "heldout.json").exists():
        stage = "scored"
    elif (res / "export.json").exists():
        stage = "scoring"
    elif history:
        stage = "round %d/%d" % (min(rounds_done + 1, n_rounds), n_rounds)
    elif res.exists():
        stage = "seed fits"
    else:
        stage = "-"
    export = read_json(res / "export.json")
    best = (export or {}).get("best_model") or (history[-1]["best_model"] if history else "-")
    counts = {}
    ledger = res / "attempted_hypotheses.jsonl"
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            if line.strip():
                outcome = json.loads(line).get("outcome", "?")
                counts[outcome] = counts.get(outcome, 0) + 1
    tokens, cost, calls, missing = 0, 0.0, 0, 0
    usage = res / "token_usage.jsonl"
    if usage.exists():
        for line in usage.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            calls += 1
            missing += bool(r.get("usage_missing"))
            tokens += sum(int(r.get(k) or 0) for k in ("input_tokens", "output_tokens", "reasoning_tokens",
                                                        "cache_read_tokens", "cache_write_tokens"))
            cost += float(r.get("cost_usd") or 0.0)
    rec = read_json(res / "recovery" / "recovery.json")
    recovered = "-" if rec is None else ("yes" if rec.get("recovered") else "no")
    rows.append([task, cell, state, stage, best,
                 "%d/%d/%d" % (counts.get("admitted", 0), counts.get("rejected", 0),
                               counts.get("pruned", 0) + counts.get("dropped", 0)),
                 "%.1fM" % (tokens / 1e6), "$%.2f%s" % (cost, "+" if missing else ""),
                 str(calls), recovered,
                 " ".join(last[1:]) if state in ("failed", "stopped") else ""])

head = ["task", "cell", "state", "stage", "best model", "adm/rej/pruned", "tokens", "cost", "agents", "recovered", "last exit"]
widths = [max(len(str(r[i])) for r in rows + [head]) for i in range(len(head))]
for r in [head] + rows:
    print("  ".join(str(v).ljust(w) for v, w in zip(r, widths)).rstrip())
print("cost '+': some agent calls reported no usage (the total is an undercount).")
PY
