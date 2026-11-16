"""Write the de-identified trial data of every live reference-game study (src/rsa/share.py). Sherlock only.

    python scripts/rsa/share_trials.py --out <dir> [--private ~/.auto-psych-private]

Reads each study's `private/raw_collected/experiment<n>.json` and its pseudonym map, and writes
`<out>/trials/<study>.csv` (one row per jsPsych record), `<out>/people.csv` (one row per submission:
study, experiment, shared id, the run's pseudonym, kept by the analysis or not, list, submission time)
and `<out>/MANIFEST.json` (counts). The secret key (`<private>/share_key`, made on first use) and the
map from Prolific id to shared id (`<private>/prolific_map.json`) are written to the private directory
and never to `<out>`. Every output table is checked for the studies' real identifiers before it is
written.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import secrets
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.rsa.live.collect import decode_trials  # noqa: E402
from src.rsa.share import (PERSON_KEEP, check_no_ids, deidentify_trials, person_hash, pseudonym_of,  # noqa: E402
                           submission_ids)

S = Path(os.path.expandvars("$SCRATCH/auto-psych"))
OPEN_V2 = S / "rsa_open_v2"
OPEN_PILOT = S / "rsa_open_live"


def run_tree(work_root: Path, cell: str) -> Path:
    return S / "agent_trees" / (work_root / "cells" / cell / "agent_tree_id").read_text().strip() / "repo/_runs/chain_0"


def closed(name, cell: Path, run: Optional[Path] = None, modelled: bool = True):
    return {"name": name, "raw": cell / "private/raw_collected", "map": cell / "private/participant_ids.json",
            "kept": "modelled" if modelled else None, "run": run or cell / "outputs"}


def open_(name, work_root: Path, cell: str):
    return {"name": name, "raw": work_root / "cells" / cell / "private/raw_collected",
            "map": work_root / "private/campaign/participant_ids.json", "kept": "rows", "run": run_tree(work_root, cell)}


def studies() -> List[dict]:
    c = S / "rsa_campaign/cells"
    return [closed("closed_pilot", S / "rsa_live_pilot/cells/live_rsa_c0_pilot", modelled=False),
            *[closed(f"closed_c{k}", c / f"live_rsa_c{k}") for k in range(3)],
            closed("closed_final", S / "rsa_prereg/cells/prereg", run=S / "rsa_prereg/cells/prereg/run"),
            open_("open_pilot1", OPEN_PILOT, "open_chain_0"),
            open_("s1_open", OPEN_V2, "open_chain_0"),
            open_("s1_fixed", OPEN_V2, "fixed_chain_0")]


def kept_pseudonyms(study: dict, experiment: str) -> Optional[Set[str]]:
    """The run's pseudonyms of the people its analysis kept, or None where the run modelled nothing."""
    data = study["run"] / experiment / "data"
    if study["kept"] == "modelled":
        with open(data / "modelled.csv", newline="") as f:
            return {r["participant_id"].split(":")[-1] for r in csv.DictReader(f)}
    if study["kept"] == "rows":
        return {json.loads(line)["participant"] for line in open(data / "rows.jsonl") if line.strip()}
    return None


def key_file(private: Path) -> bytes:
    private.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = private / "share_key"
    if not path.exists():
        path.write_text(secrets.token_hex(32))
        path.chmod(0o600)
    return bytes.fromhex(path.read_text().strip())


def write_csv(path: Path, rows: List[dict], first: List[str]) -> None:
    cols = list(dict.fromkeys(first + [k for r in rows for k in r]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--private", type=Path, default=Path.home() / ".auto-psych-private")
    args = ap.parse_args(argv)
    key = key_file(args.private)
    people: List[dict] = []
    prolific_map: Dict[str, dict] = {}
    manifest = {"studies": {}}
    all_ids: Set[str] = set()
    for st in studies():
        id_map = json.loads(st["map"].read_text())
        trials: List[dict] = []
        for raw in sorted(st["raw"].glob("experiment*.json")):
            experiment = raw.stem
            kept = kept_pseudonyms(st, experiment)
            responses = json.loads(raw.read_text())
            seen_pseudonyms = set()
            for resp in responses:
                records = decode_trials(resp["trials"])
                pid = resp["prolific_pid"]
                ids = submission_ids(resp, records)
                all_ids |= ids
                if any(r.get("participant_id") not in (None, pid) for r in records):
                    raise ValueError(f"{st['name']} {experiment}: a submission's records name another participant")
                h = person_hash(key, pid)
                pseudo = pseudonym_of(id_map, pid)
                seen_pseudonyms.add(pseudo)
                trials += [{"study": st["name"], "experiment": experiment, **r} for r in deidentify_trials(records, h)]
                people.append({"study": st["name"], "experiment": experiment, "person": h, "run_pseudonym": pseudo or "",
                               "kept": "" if kept is None else pseudo in kept, "n_records": len(records),
                               **{k: resp.get(k) for k in PERSON_KEEP}})
                entry = prolific_map.setdefault(pid, {"person": h, "studies": []})
                entry["studies"].append(f"{st['name']}/{experiment}")
            if kept is not None and not kept <= seen_pseudonyms:
                raise ValueError(f"{st['name']} {experiment}: {len(kept - seen_pseudonyms)} kept people have no "
                                 "stored submission")
            manifest["studies"].setdefault(st["name"], {})[experiment] = {
                "submissions": len(responses), "kept": None if kept is None else len(kept)}
        check_no_ids(trials, all_ids)
        write_csv(args.out / "trials" / f"{st['name']}.csv", trials, ["study", "experiment", "person"])
        manifest["studies"][st["name"]]["trial_rows"] = len(trials)
        print(f"[share] {st['name']}: {len(trials)} trial rows, "
              f"{sum(1 for p in people if p['study'] == st['name'])} submissions", flush=True)
    check_no_ids(people, all_ids)
    write_csv(args.out / "people.csv", people, ["study", "experiment", "person", "run_pseudonym", "kept"])
    manifest["people"] = len({p["person"] for p in people})
    manifest["submissions"] = len(people)
    (args.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    mp = args.private / "prolific_map.json"
    mp.write_text(json.dumps(prolific_map, indent=1))
    mp.chmod(0o600)
    print(f"[share] {manifest['submissions']} submissions from {manifest['people']} people -> {args.out}; "
          f"key and Prolific map in {args.private}")


if __name__ == "__main__":
    main()
