"""A live collection that gives up before its target pauses the study.

After three hours (``_PROLIFIC_MAX_WAIT_SEC``) without enough completed
submissions the poll moves on and the pipeline models the partial data — but
it left the Prolific study recruiting, so people kept being recruited and paid
for data no experiment would use.
Now the study is paused (a reversible transition: START resumes it) and a
failure to pause stops the run loudly. A collection that reaches its target
leaves the study alone: its places are all taken, so it recruits no one else.

Prolific is faked at the HTTP layer of ``src.runtime.prolific``; nothing is
contacted.
"""

from __future__ import annotations

import io

import pytest

import src.runtime.prolific as prolific_client
from src.pipelines.outer_loop import collect

STUDY = "study-live-1"
RESULTS_CSV = (
    "participant_id,trial_index,sequence_a,sequence_b,chose_left\n"
    "0,0,HHT,THT,1\n"
    "0,1,HTHT,HHHH,0\n"
)


class _Response:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


class FakeProlific:
    """A study with ``completed`` finished submissions and a ``status``."""

    def __init__(self, *, completed, status="ACTIVE", transition_status=200):
        self.completed = completed
        self.status = status
        self.transition_status = transition_status
        self.transitions: list[str] = []

    def get(self, url, **kwargs):
        if url.endswith(f"/studies/{STUDY}/submissions/counts/"):
            return _Response(200, {"APPROVED": self.completed})
        if url.endswith(f"/studies/{STUDY}/"):
            return _Response(200, {"id": STUDY, "status": self.status})
        raise AssertionError(f"unexpected GET {url}")

    def post(self, url, json=None, **kwargs):
        assert url.endswith(f"/studies/{STUDY}/transition/"), url
        self.transitions.append(json["action"])
        if self.transition_status == 200 and json["action"] == "PAUSE":
            self.status = "PAUSED"
        return _Response(self.transition_status, {"status": self.status})


class _Clock:
    """time.monotonic/time.sleep stand-in: sleeping advances the clock."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


@pytest.fixture
def run_collection(tmp_path, monkeypatch):
    def run(prolific: FakeProlific, target: int = 3):
        monkeypatch.setattr(prolific_client, "_headers", lambda: {"Authorization": "Token x"})
        monkeypatch.setattr(prolific_client.requests, "get", prolific.get)
        monkeypatch.setattr(prolific_client.requests, "post", prolific.post)
        monkeypatch.setattr(collect, "time", _Clock())
        monkeypatch.setattr(
            collect.urllib.request, "urlopen",
            lambda *a, **k: io.BytesIO(RESULTS_CSV.encode("utf-8")),
        )
        monkeypatch.setenv("AUTO_PSYCH_RESULTS_TOKEN", "token")
        return collect._collect_live(
            {"project_id": "subjective_randomness", "run_id": 1},
            {
                "prolific_study_id": STUDY,
                "results_api_url": "http://results.invalid",
                "total_available_places": target,
            },
            tmp_path,
            tmp_path,
        )

    return run


def test_a_timed_out_collection_pauses_the_study_and_models_the_partial_data(
    run_collection, capsys
):
    prolific = FakeProlific(completed=1)

    rows = run_collection(prolific)

    assert prolific.transitions == ["PAUSE"]
    assert len(rows) == 2
    out = capsys.readouterr().out
    assert f"PAUSED Prolific study {STUDY}" in out
    assert "1/3" in out


def test_a_failed_pause_stops_the_run(run_collection):
    prolific = FakeProlific(completed=1, transition_status=500)

    with pytest.raises(RuntimeError, match=f"Could not pause Prolific study {STUDY}"):
        run_collection(prolific)


def test_a_study_that_already_stopped_recruiting_is_left_as_is(run_collection, capsys):
    # E.g. a RESUME_AGENTS=4_collect re-run after the first attempt paused it.
    prolific = FakeProlific(completed=1, status="PAUSED")

    run_collection(prolific)

    assert prolific.transitions == []
    assert "already PAUSED" in capsys.readouterr().out


def test_a_study_in_an_unexpected_state_stops_the_run(run_collection):
    prolific = FakeProlific(completed=1, status="SCHEDULED")

    with pytest.raises(RuntimeError, match="SCHEDULED"):
        run_collection(prolific)


def test_a_collection_that_reached_its_target_leaves_the_study_alone(run_collection):
    prolific = FakeProlific(completed=3)

    rows = run_collection(prolific)

    assert prolific.transitions == []
    assert len(rows) == 2
