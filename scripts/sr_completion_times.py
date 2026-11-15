"""Completion times for the sr-oct26 subjective-randomness live studies.

Reads the nine Prolific study IDs from the shared study_ids.csv, fetches each
study's submissions from the Prolific API, and prints completion-time summaries
(minutes) per study, per run and pooled. Prints no participant IDs.

Needs only `requests`. Run it with the token of the account that owns the
studies:

    PROLIFIC_API_TOKEN=... python sr_completion_times.py
"""

import csv
import os
import statistics

import requests

STUDY_IDS_CSV = "/scratch/groups/mcfrank/sr_data/study_ids.csv"
API = "https://api.prolific.com/api/v1"
# Submissions that finished the task; returned / timed-out ones are excluded.
COMPLETED_STATUSES = {"APPROVED", "AWAITING REVIEW"}


def fetch_submissions(study_id, token):
    response = requests.get(
        f"{API}/studies/{study_id}/submissions/",
        headers={"Authorization": f"Token {token}"},
        params={"limit": 500},
        timeout=60,
    )
    response.raise_for_status()
    body = response.json()
    submissions = body["results"]
    if len(submissions) != body["meta"]["count"]:
        raise RuntimeError(
            f"study {study_id}: got {len(submissions)} of "
            f"{body['meta']['count']} submissions in one page"
        )
    return submissions


def summarize(label, minutes):
    print(
        f"{label:<22} n={len(minutes):>3}  "
        f"mean={statistics.mean(minutes):5.2f}  "
        f"sd={statistics.stdev(minutes):5.2f}  "
        f"median={statistics.median(minutes):5.2f}  "
        f"min={min(minutes):5.2f}  max={max(minutes):6.2f}"
    )


def main():
    token = os.environ["PROLIFIC_API_TOKEN"]
    with open(STUDY_IDS_CSV) as f:
        studies = list(csv.DictReader(f))

    minutes_by_run = {}
    print("Completion time (minutes), completed submissions only\n")
    for study in studies:
        submissions = fetch_submissions(study["prolific_study_id"], token)
        status_counts = {}
        for s in submissions:
            status_counts[s["status"]] = status_counts.get(s["status"], 0) + 1
        minutes = [
            s["time_taken"] / 60
            for s in submissions
            if s["status"] in COMPLETED_STATUSES
        ]
        summarize(f"{study['run']} {study['experiment']}", minutes)
        print(f"{'':<22} statuses: {status_counts}")
        minutes_by_run.setdefault(study["run"], []).extend(minutes)

    print()
    for run, minutes in minutes_by_run.items():
        summarize(f"{run} (all experiments)", minutes)
    summarize("all runs", [m for ms in minutes_by_run.values() for m in ms])


if __name__ == "__main__":
    main()
