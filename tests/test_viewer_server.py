"""Integration tests for the run-explorer web server.

The viewer walks the data tree on demand and exposes each run — its experiment
units, and each unit's full payload (theory, design, data, model loop with
candidates per step, and critiques with their PPC results). These outside-in
tests drive the public HTTP surface.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.viewer.server import create_app
from tests.viewer_fixtures import build_demo_tree


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    return build_demo_tree(tmp_path / "data")


@pytest.fixture
def client(data_root: Path):
    app = create_app(data_root=data_root)
    app.config.update(TESTING=True)
    return app.test_client()


def test_index_lists_every_run(client):
    resp = client.get("/api/index")
    assert resp.status_code == 200
    paths = {r["path"] for r in resp.get_json()["runs"]}
    assert paths == {
        "outer_loop/demo",
        "recovery/holdout_runs/cond_a",
        "recovery/confusion_runs/model_x",
        "thinkaloud",
    }


def test_run_summary(client):
    run = client.get("/api/run/outer_loop/demo").get_json()
    assert run["label"] == "demo"
    assert {e["unit"] for e in run["experiments"]} == {"experiment1", "smoke"}
    assert run["figures"] == ["analysis/loop_trajectory.png"]


def test_run_experiment_full_payload(client):
    exp = client.get("/api/run/outer_loop/demo/experiment?unit=experiment1").get_json()
    assert exp["data"]["n_participants"] == 2
    assert exp["model_loop"]["candidates"][0]["name"] == "iter0_candidate0"
    crit = next(c for c in exp["critiques"] if c["iteration"] == 0)
    sig = next(s for s in crit["stats"] if s["name"] == "alternation_rate_gap")
    assert sig["significant"] is True
    assert sig["p_value_adjusted"] == pytest.approx(0.024)


def test_run_experiment_unit_param_defaults_and_partial(client):
    exp = client.get("/api/run/outer_loop/demo/experiment?unit=smoke").get_json()
    assert exp["model_loop"] is None
    assert exp["theory"]["models"] == []


def test_nested_run_experiment(client):
    exp = client.get(
        "/api/run/recovery/holdout_runs/cond_a/experiment?unit=experiment1"
    ).get_json()
    assert any(c["name"] == "iter0_candidate0" for c in exp["model_loop"]["candidates"])


def test_run_file_serves_figure(client):
    resp = client.get("/api/run/outer_loop/demo/files/analysis/loop_trajectory.png")
    assert resp.status_code == 200
    assert resp.data[:4] == b"\x89PNG"


def test_unknown_run_is_404(client):
    assert client.get("/api/run/outer_loop/nope").status_code == 404
    assert client.get("/api/run/outer_loop/demo/experiment?unit=nope").status_code == 404


def test_root_serves_html_shell(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"<html" in resp.data.lower()
