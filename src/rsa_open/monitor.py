"""A running open-campaign study at a glance (`src.rsa_open.live.monitor_report`).

    python -m src.rsa_open.monitor --exp-dir <run>/experiment<n> [--json]

Reads the experiment's deployment manifest and frozen trial lists, fetches the study's stored
responses from ``/results`` (needs ``AUTO_PSYCH_RESULTS_TOKEN``) and prints: responses and
exclusions by reason, comprehension and catch-trial rates, people who gave the same answer on
nearly every test trial (for every response type: `src.rsa_open.live.answer_key`), people who
answered implausibly fast, completion times, and the pay per hour at the median time (a warning
near Prolific's $8 floor). People are numbered here (``r<k>``, in order of
submission); nothing written or printed carries a Prolific id. The chain's own job prints the
same report every 10 minutes while a study fills.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import tyro


@dataclass
class Args:
    exp_dir: Path
    json: bool = False


def main(args: Args) -> dict:
    from src.pipelines.outer_loop.deployment.manifest import manifest_path
    from src.rsa.live.collect import fetch_responses
    from src.rsa_open.live import _pids, exclusions, monitor_lines, monitor_report, read_people
    from src.rsa_open.paradigm import load_paradigm

    manifest = json.loads(manifest_path(args.exp_dir).read_text())
    lists = json.loads((Path(args.exp_dir) / "design" / "frozen" / "lists.json").read_text())
    study = manifest.get("prolific_study_id")
    if not study:
        raise SystemExit(f"{args.exp_dir}: the deployment records no Prolific study")
    responses = fetch_responses(manifest)
    pids = _pids(responses, study)
    people, skipped = read_people(responses, study, lists, {pid: f"r{i}" for i, pid in enumerate(pids)})
    pay = load_paradigm(Path(args.exp_dir) / "design" / "frozen" / "paradigm.json").design.pay_usd
    report = {**monitor_report(people, {p.id: w for p in people if (w := exclusions(p, False))}, pay_usd=pay),
              **skipped}
    if args.json:
        print(json.dumps(report, indent=1))
    else:
        print(f"study {study} ({manifest.get('prolific_mode')}), {manifest.get('experiment_url')}")
        print("\n".join(monitor_lines(report)))
    return report


if __name__ == "__main__":
    main(tyro.cli(Args))
