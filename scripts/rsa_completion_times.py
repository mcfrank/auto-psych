"""Completion times for the live RSA reference-game studies (chains c0-c2 and the prereg).

For each run's private directory, reads the run directory from the recorded
config, the Prolific study ID from each experiment's deployment manifest, and
the map from Prolific IDs to the run's participant IDs (``live<k>``). Fetches
each study's submissions from the Prolific API, writes one row per submission
with the Prolific IDs replaced by ``live<k>``, and prints mean and median
completion times (minutes) per study, per run and pooled.

Run it with the token of the account that owns the studies:

    PROLIFIC_API_TOKEN=... uv run python scripts/rsa_completion_times.py
"""

import csv
import json
import os
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import requests
import tyro
from pyprojroot import here

API = "https://api.prolific.com/api/v1"
# Submissions that finished the task; returned / timed-out ones are excluded from the summaries.
COMPLETED_STATUSES = {"APPROVED", "AWAITING REVIEW"}
CELLS = Path("/scratch/users/mcfrank/auto-psych")


@dataclass
class Args:
    private_dirs: list[Path] = field(default_factory=lambda: [
        CELLS / "rsa_campaign/cells/live_rsa_c0/private",
        CELLS / "rsa_campaign/cells/live_rsa_c1/private",
        CELLS / "rsa_campaign/cells/live_rsa_c2/private",
        CELLS / "rsa_prereg/cells/prereg/private",
    ])
    """Each run's private directory (holds outer_config.json or prereg_config.json and participant_ids.json)."""
    output: Path = here("data/rsa/live/completion_times/rsa_completion_times.csv")
    """Where to write the per-submission table (no Prolific IDs)."""


def read_run_config(private_dir):
    for name in ("outer_config.json", "prereg_config.json"):
        path = private_dir / name
        if path.exists():
            return json.loads(path.read_text())
    raise FileNotFoundError(f"{private_dir} has neither outer_config.json nor prereg_config.json")


def study_ids_by_experiment(run_dir):
    """The Prolific study ID recorded in each experiment's deployment manifest."""
    manifests = sorted(Path(run_dir).glob("experiment*/deployment/deployment_manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"no deployment manifests under {run_dir}")
    study_ids = {}
    for path in manifests:
        study_id = json.loads(path.read_text())["prolific_study_id"]
        if not study_id:
            raise ValueError(f"{path} records no Prolific study")
        study_ids[path.parent.parent.name] = study_id
    return study_ids


def fetch_submissions(study_id, token):
    """Every submission of the study, following the API's pagination."""
    url = f"{API}/studies/{study_id}/submissions/"
    params = {"limit": 500}
    submissions = []
    while url:
        response = requests.get(url, headers={"Authorization": f"Token {token}"}, params=params, timeout=60)
        response.raise_for_status()
        body = response.json()
        submissions.extend(body["results"])
        expected = body["meta"]["count"]
        url = (body.get("_links", {}).get("next") or {}).get("href")
        params = None  # the next link carries its own query string
    if len(submissions) != expected:
        raise RuntimeError(f"study {study_id}: got {len(submissions)} of {expected} submissions")
    return submissions


def summarize(label, minutes):
    print(
        f"{label:<24} n={len(minutes):>4}  "
        f"mean={statistics.mean(minutes):5.2f}  "
        f"median={statistics.median(minutes):5.2f}  "
        f"sd={statistics.stdev(minutes):5.2f}  "
        f"min={min(minutes):5.2f}  max={max(minutes):6.2f}"
    )


def main(args: Args):
    token = os.environ["PROLIFIC_API_TOKEN"]

    rows = []
    for private_dir in args.private_dirs:
        config = read_run_config(private_dir)
        run = config["run_label"]
        # Prolific ID -> live<k>; people who returned without data are not in the map.
        participant_ids = json.loads((private_dir / "participant_ids.json").read_text())
        for experiment, study_id in study_ids_by_experiment(config["run_dir"]).items():
            for submission in fetch_submissions(study_id, token):
                rows.append({
                    "run": run,
                    "experiment": experiment,
                    "participant_id": participant_ids.get(submission["participant_id"], ""),
                    "status": submission["status"],
                    "time_taken_seconds": submission["time_taken"],
                })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} submissions to {args.output}\n")

    completed = [r for r in rows if r["status"] in COMPLETED_STATUSES]
    print("Completion time (minutes), completed submissions only\n")
    for run in dict.fromkeys(r["run"] for r in completed):
        run_rows = [r for r in completed if r["run"] == run]
        for experiment in dict.fromkeys(r["experiment"] for r in run_rows):
            summarize(f"{run} {experiment}",
                      [r["time_taken_seconds"] / 60 for r in run_rows if r["experiment"] == experiment])
        summarize(f"{run} (all experiments)", [r["time_taken_seconds"] / 60 for r in run_rows])
    print()
    summarize("all runs", [r["time_taken_seconds"] / 60 for r in completed])


if __name__ == "__main__":
    main(tyro.cli(Args))
