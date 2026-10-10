"""Deploy an RSA experiment, recruit on Prolific, collect its data (the live outer loop).

Main's machinery does the deployment and the recruiting
(`src.pipelines.outer_loop.deployment.local.run_deployment`; `collect.py`'s
Prolific polling and pause). This module adds what is specific to the
reference game:

* the page (`src.rsa.live.site`): server-assigned lists, a post to /submit;
* the data, read whole from ``/results?format=json`` (the CSV export knows only
  main's columns) and converted by `src.rsa.experiment.convert`, so live rows
  are what simulated rows are;
* participant ids: ``live<k>``, numbered across the run's experiments, from a
  private map of Prolific ids. The raw responses, which carry the Prolific
  ids, go to the private directory, never to the run tree the agents read.

Every failure raises: a study that ran must never read as "nobody responded".
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.pipelines.outer_loop.collect import (
    _PROLIFIC_MAX_WAIT_SEC,
    _pause_unfilled_study,
    _poll_prolific_until_target,
    _results_request,
)
from src.pipelines.outer_loop.deployment.manifest import manifest_path, recorded_live_study
from src.rsa.experiment.convert import convert
from src.rsa.live.site import build_live_site

PROJECT_ID = "rsa_reference"  # the Prolific config's project (src/pipelines/outer_loop/projects/)


TRIALS_FORMAT = "rsa_jspsych_json"  # how the page stores its data in Firestore (src.rsa.live.site)


def decode_trials(stored) -> List[dict]:
    """The page's jsPsych records from a stored response's ``trials``: one
    JSON string (Firestore rejects nested arrays, and each trial's display is
    a matrix). Anything else is not this page's data, and raises."""
    if (isinstance(stored, list) and len(stored) == 1 and isinstance(stored[0], dict)
            and stored[0].get("format") == TRIALS_FORMAT and isinstance(stored[0].get("data"), str)):
        records = json.loads(stored[0]["data"])
        if not isinstance(records, list):
            raise ValueError("the stored jsPsych data are not a list of records")
        return records
    raise ValueError(f"stored trials are not {TRIALS_FORMAT!r} (one JSON string): another page's data?")


class DraftOnly(RuntimeError):
    """A test-mode deployment: the page is live and the study is a Prolific draft, nothing to collect."""


@dataclass
class LiveSettings:
    prolific_mode: str  # "test" (a draft to preview) or "live" (published; real money)
    firebase_project: Optional[str]
    run_label: str  # this run's label on its Hosting site and study (e.g. "c0")
    collection_owner: str
    repo_root: Path
    firebase_region: str = "us-central1"
    max_wait_sec: float = _PROLIFIC_MAX_WAIT_SEC


def deploy(exp_dir: Path, lists_doc: dict, n_participants: int, experiment: int, s: LiveSettings) -> dict:
    """The experiment's deployment manifest: the recorded live study if there is
    one (a resumed run never deploys or publishes twice), else a new deployment."""
    from src.pipelines.outer_loop.deployment.local import run_deployment

    if s.prolific_mode not in ("test", "live"):
        raise ValueError(f"prolific_mode must be 'test' or 'live', not {s.prolific_mode!r}")
    if recorded_live_study(exp_dir) is None:
        if len(lists_doc["lists"]) != n_participants:
            raise ValueError(f"{len(lists_doc['lists'])} trial lists for {n_participants} participants")
        build_live_site(lists_doc, exp_dir, overwrite=True)
        run_deployment(exp_dir=exp_dir, project_id=PROJECT_ID, run_id=experiment, deploy_target="firebase",
                       prolific_mode=s.prolific_mode, agent_backend="opencode", collection_owner=s.collection_owner,
                       firebase_project=s.firebase_project, firebase_region=s.firebase_region,
                       n_participants=n_participants, repo_root=s.repo_root, run_label=s.run_label)
    manifest = json.loads(manifest_path(exp_dir).read_text())
    if s.prolific_mode == "test":
        raise DraftOnly(f"test mode: the page is at {manifest.get('experiment_url')} and Prolific study "
                        f"{manifest.get('prolific_study_id')} is a draft to preview; nothing to collect")
    return manifest


def wait_for_participants(manifest: dict, target: int, log_dir: Path, max_wait_sec: float) -> int:
    """Poll Prolific until ``target`` submissions or the wait ends; pause a study left short."""
    study = manifest.get("prolific_study_id")
    if not study:
        raise RuntimeError("the deployment recorded no Prolific study: nothing to collect from")
    completed = _poll_prolific_until_target(study, target, log_dir, max_wait_sec=max_wait_sec)
    if completed < target:
        _pause_unfilled_study(study, completed, target, log_dir)
    return completed


def fetch_responses(manifest: dict) -> List[dict]:
    """Every stored response of the deployment's collection session, whole."""
    base = manifest.get("results_api_url")
    session = manifest.get("collection_session_id")
    if not base or not session:
        raise RuntimeError("the deployment manifest has no results_api_url or collection_session_id")
    url = f"{base.rstrip('/')}/results?collection_session_id={session}&format=json"
    try:
        with urllib.request.urlopen(_results_request(url), timeout=120) as r:
            body = json.loads(r.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"live results fetch failed for {url}: {type(exc).__name__}: {exc}. The participants' "
                           "data are on the server: refusing to report zero responses.") from exc
    if not isinstance(body, list):
        raise RuntimeError(f"{url} returned {type(body).__name__}, not a list of responses")
    return body


def responses_to_rows(responses: List[dict], *, study_id: str, experiment: str,
                      ids: Dict[str, str]) -> tuple[pd.DataFrame, Dict[str, str], dict]:
    """Canonical rows for the study's responses. ``ids`` maps Prolific ids to the
    run's participant ids (extended here: new people get the next ``live<k>``).
    Responses of another study (a preview, a test submission) are left out, on record."""
    ids = dict(ids)
    frames, other, empty = [], [], []
    for resp in sorted(responses, key=lambda r: (r.get("submitted_at_client") or "", r["participant_id_str"])):
        if resp.get("prolific_study_id") != study_id:
            other.append(resp["participant_id_str"])
            continue
        pid = resp.get("prolific_pid") or resp["participant_id_str"]
        if pid not in ids:
            ids[pid] = f"live{len(ids)}"
        frame = convert(decode_trials(resp["trials"]), participant_id=ids[pid], experiment=experiment)
        if frame.empty:
            empty.append(ids[pid])
            continue
        frames.append(frame)
    if not frames:
        raise RuntimeError(f"no response of study {study_id} has any trial ({len(responses)} responses stored)")
    record = dict(n_responses=len(frames), other_study=len(other), without_trials=empty)
    return pd.concat(frames, ignore_index=True), ids, record
