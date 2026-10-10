"""Live collection in the RSA outer loop, with the deployment and Prolific stubbed."""

import json

import pandas as pd
import pytest

from src.rsa.experiment.design import EXPERIMENT_ASSETS_DIR, Design, trial_lists
from src.rsa.live import collect as live
from src.rsa.outer.run import OuterConfig, OuterRun
from src.rsa.outer.simulate import page_records


def _setup(tmp_path, monkeypatch, *, prolific_mode="live", confirm=True, responses=None):
    doc = trial_lists(Design.load(EXPERIMENT_ASSETS_DIR / "demo_design.json"), seed=3, n_lists=4, n_catch=2)
    lists_path = tmp_path / "lists.json"
    lists_path.write_text(json.dumps(doc))
    cfg = OuterConfig(run_dir=tmp_path / "run", seeds=tmp_path, existing_data=tmp_path / "e.csv",
                      private_dir=tmp_path / "private", collection="live", participants=4, prolific_mode=prolific_mode,
                      confirm_live_recruitment=confirm, run_label="c0", firebase_project="proj", max_catch_errors=2)
    run = OuterRun(cfg, spawner=None)
    monkeypatch.setattr(run, "design", lambda n: lists_path)
    calls = {}

    def fake_deploy(exp_dir, lists_doc, n_participants, experiment, s):
        calls["deploy"] = (n_participants, experiment, s.prolific_mode, s.run_label)
        return {"prolific_study_id": "STUDY", "collection_session_id": "cs", "results_api_url": "https://x"}

    monkeypatch.setattr(live, "deploy", fake_deploy)
    monkeypatch.setattr(live, "wait_for_participants", lambda m, target, log_dir, wait: target)
    if responses is None:
        responses = [dict(participant_id_str=f"PID{k}", prolific_pid=f"PID{k}", prolific_study_id="STUDY",
                          submitted_at_client=f"2026-10-10T10:0{k}:00Z", list_index=k,
                          trials=page_records(doc, k, [0] * len(doc["lists"][k]["trials"]), participant_id=f"PID{k}"))
                     for k in range(4)]
        responses.append(dict(participant_id_str="PREVIEW", prolific_pid=None, prolific_study_id="other",
                              trials=responses[0]["trials"]))
    monkeypatch.setattr(live, "fetch_responses", lambda m: responses)
    return run, doc, calls


def test_live_responses_become_the_experiments_rows_with_private_prolific_ids(tmp_path, monkeypatch):
    run, doc, calls = _setup(tmp_path, monkeypatch)
    rows = pd.read_csv(run.collect(1))
    assert calls["deploy"] == (4, 1, "live", "c0")
    assert sorted(rows["participant_id"].unique()) == [f"auto_psych:live{k}" for k in range(4)]
    assert set(rows["experiment"]) == {"run_e1"} and set(rows["source"]) == {"auto_psych"}
    n_tests = sum(t["phase"] == "test" and not t["is_catch"] for t in doc["lists"][0]["trials"])
    assert len(rows[rows["condition"] != "catch"]) == 4 * n_tests
    # Nothing an agent can read carries a Prolific id; the private dir keeps them.
    visible = "".join(p.read_text() for p in (tmp_path / "run").rglob("*") if p.is_file())
    assert "PID0" not in visible and "PREVIEW" not in visible
    raw = json.loads((tmp_path / "private" / "raw_collected" / "experiment1.json").read_text())
    assert {r["participant_id_str"] for r in raw} == {"PID0", "PID1", "PID2", "PID3", "PREVIEW"}
    assert json.loads((tmp_path / "private" / "participant_ids.json").read_text())["PID2"] == "live2"
    record = json.loads((tmp_path / "run" / "experiment1" / "data" / "participants.json").read_text())
    assert record["n_recruited"] == 4 and record["other_study"] == 1 and record["n_target"] == 4


def test_a_published_study_needs_the_confirmation_too(tmp_path, monkeypatch):
    run, _, calls = _setup(tmp_path, monkeypatch, confirm=False)
    with pytest.raises(RuntimeError, match="confirm-live-recruitment"):
        run.collect(1)
    assert "deploy" not in calls


def test_ids_continue_across_experiments_and_a_returning_person_keeps_theirs():
    doc = trial_lists(Design.load(EXPERIMENT_ASSETS_DIR / "demo_design.json"), seed=3, n_lists=2, n_catch=0)
    resp = lambda pid, k: dict(participant_id_str=pid, prolific_pid=pid, prolific_study_id="S2",  # noqa: E731
                               trials=page_records(doc, k, [0] * len(doc["lists"][k]["trials"])))
    _, ids, _ = live.responses_to_rows([resp("A", 0), resp("NEW", 1)], study_id="S2", experiment="r_e2",
                                       ids={"A": "live0", "B": "live1"})
    assert ids == {"A": "live0", "B": "live1", "NEW": "live2"}


def test_a_study_with_no_usable_response_raises():
    with pytest.raises(RuntimeError, match="no response of study"):
        live.responses_to_rows([dict(participant_id_str="X", prolific_study_id="other", trials=[])],
                               study_id="S", experiment="e", ids={})


def test_a_test_mode_deployment_stops_the_run_after_the_draft(tmp_path, monkeypatch):
    run, _, calls = _setup(tmp_path, monkeypatch, prolific_mode="test", confirm=False)

    def draft(exp_dir, lists_doc, n, experiment, s):
        calls["deploy"] = s.prolific_mode
        raise live.DraftOnly("a draft to preview")

    monkeypatch.setattr(live, "deploy", draft)
    run.run()
    assert calls["deploy"] == "test" and not (tmp_path / "run" / "experiment1" / "data" / "responses.csv").exists()
    # Going live afterwards is the same run: only how it recruits changed.
    run.cfg.prolific_mode, run.cfg.confirm_live_recruitment = "live", True
    monkeypatch.setattr(run, "prospective", lambda n: (_ for _ in ()).throw(StopIteration("past collection")))
    monkeypatch.setattr(live, "deploy", lambda *a: {"prolific_study_id": "STUDY", "collection_session_id": "cs",
                                                   "results_api_url": "https://x"})
    with pytest.raises(StopIteration, match="past collection"):
        run.run()
    assert (tmp_path / "run" / "experiment1" / "data" / "responses.csv").exists()


def test_deploy_builds_the_page_and_never_deploys_over_a_recorded_live_study(tmp_path, monkeypatch):
    import src.pipelines.outer_loop.deployment.local as local
    from src.pipelines.outer_loop.deployment.manifest import manifest_path

    doc = trial_lists(Design.load(EXPERIMENT_ASSETS_DIR / "demo_design.json"), seed=3, n_lists=4, n_catch=2)
    exp = tmp_path / "experiment1"
    deployed = []

    def fake_run_deployment(**kw):
        deployed.append(kw)
        assert (kw["exp_dir"] / "experiment" / "index.html").exists()
        manifest_path(kw["exp_dir"]).parent.mkdir(parents=True, exist_ok=True)
        manifest_path(kw["exp_dir"]).write_text(json.dumps(
            {"prolific_mode": kw["prolific_mode"], "prolific_study_id": "S1", "experiment_url": "https://p/e1-c0/"}))

    monkeypatch.setattr(local, "run_deployment", fake_run_deployment)
    s = live.LiveSettings(prolific_mode="test", firebase_project="proj", run_label="c0", collection_owner="me",
                          repo_root=tmp_path)
    with pytest.raises(live.DraftOnly, match="draft"):
        live.deploy(exp, doc, 4, 1, s)
    assert deployed[0]["prolific_mode"] == "test" and deployed[0]["run_label"] == "c0"
    assert deployed[0]["project_id"] == "rsa_reference" and deployed[0]["n_participants"] == 4
    s.prolific_mode = "live"
    monkeypatch.setattr(live, "recorded_live_study", lambda d: object())  # a live study is on record
    assert live.deploy(exp, doc, 4, 1, s)["prolific_study_id"] == "S1" and len(deployed) == 1
    with pytest.raises(ValueError, match="trial lists"):
        monkeypatch.setattr(live, "recorded_live_study", lambda d: None)
        live.deploy(tmp_path / "experiment2", doc, 5, 2, s)


def test_a_pilot_stops_after_its_data_are_in(tmp_path, monkeypatch):
    run, _, _ = _setup(tmp_path, monkeypatch)
    run.cfg.stop_after_collect = True
    monkeypatch.setattr(run, "prospective", lambda n: (_ for _ in ()).throw(AssertionError("modelled a pilot")))
    run.run()
    assert (tmp_path / "run" / "experiment1" / "data" / "responses.csv").exists()
