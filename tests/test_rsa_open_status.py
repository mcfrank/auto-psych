"""The campaign's status at a glance (src/rsa_open/status.py), on a made-up work root."""

import json
import os
import time


def _chain(work_root, trees, cell, tree_id, *, stop=None, done=False):
    cdir = work_root / "cells" / cell
    cdir.mkdir(parents=True)
    (cdir / "agent_tree_id").write_text(tree_id)
    if stop:
        (cdir / "STOP").write_text(stop)
    if done:
        (cdir / "DONE").write_text("x")
    chain = cell.split("_chain_")[1].split("_")[0]
    run = trees / tree_id / "repo" / "_runs" / f"chain_{chain}"
    run.mkdir(parents=True)
    return run


def _experiment(run, n, *, loop_rounds=None, done=False, recruiting=False, kept=None):
    exp = run / f"experiment{n}"
    (exp / "design" / "frozen").mkdir(parents=True)
    (exp / "design" / "frozen" / "record.json").write_text(json.dumps({"submitted_by": "designer"}))
    (exp / "design" / "frozen" / "paradigm.json").write_text(json.dumps(
        {"response": {"type": "bet"}, "design": {"n_participants": 150, "pay_usd": 1.5}}))
    (exp / "deployment").mkdir()
    (exp / "deployment" / "deployment_manifest.json").write_text(json.dumps(
        {"prolific_mode": "live", "prolific_study_id": f"S{n}"}))
    (exp / "deployment" / "collection_started.json").write_text(json.dumps({"at": time.time() - 3600}))
    if not recruiting:
        (exp / "deployment" / "collection_ended.json").write_text(json.dumps({"completed": 150, "target": 150}))
    else:
        (exp / "deployment" / "monitor.json").write_text(json.dumps({"recruited": 40}))
    if kept is not None:
        (exp / "data").mkdir()
        (exp / "data" / "participants.json").write_text(json.dumps(
            {"recruited": 150, "kept": kept, "exclusion_rate": 0.1, "enjoyment": {"mean": 3.8}}))
    if loop_rounds is not None:
        (exp / "loop").mkdir()
        history = [{"round": r, "best_model": f"best_{r}", "live": [{"name": "a"}, {"name": "b"}]}
                   for r in range(-1, loop_rounds)]
        (exp / "loop" / "history.json").write_text(json.dumps(history))
        if done:
            (exp / "loop" / "export.json").write_text(json.dumps({"best_model": "x", "live": ["a"]}))
    return exp


def test_every_chain_its_state_and_the_studies_recruiting(tmp_path):
    from src.rsa_open.status import campaign_status, lines

    work, trees = tmp_path / "rsa_open_live", tmp_path / "agent_trees"
    run = _chain(work, trees, "open_chain_0", "aa")
    _experiment(run, 1, loop_rounds=5, done=True, kept=123)
    _experiment(run, 2, loop_rounds=2, kept=151)
    (run / "stage_times.jsonl").write_text(json.dumps({"experiment": 2, "stage": "loop", "event": "started",
                                                       "at": time.time(), "job": "1"}) + "\n")
    run2 = _chain(work, trees, "fixed_chain_1", "bb")
    _experiment(run2, 1, recruiting=True)
    _chain(work, trees, "open_chain_2", "cc", stop="2026-11-03\tawaiting the PI's review")
    report = campaign_status(work)
    by = {c["cell"]: c for c in report["chains"]}
    assert by["open_chain_0"]["state"] == "running"
    assert by["open_chain_0"]["experiments"][1]["loop"] == {"rounds_scored": 2, "best": "best_1", "done": False,
                                                            "live": 2}
    assert by["open_chain_0"]["participant_usd"] == round(2 * 150 * 1.5 * 4 / 3, 2)
    assert by["open_chain_2"]["state"] == "stopped" and "review" in by["open_chain_2"]["stop"]
    assert [(r["cell"], r["experiment"]) for r in report["recruiting"]] == [("fixed_chain_1", 1)]
    text = "\n".join(lines(report))
    assert "exp2 loop started" in text and "RECRUITING 40/?" in text and "STOP 2026-11-03" in text
    assert "loop: 2 round(s) scored, best best_1, 2 live" in text


def test_a_chain_whose_files_stopped_changing_is_flagged_quiet(tmp_path):
    from src.rsa_open.status import campaign_status

    work, trees = tmp_path / "w", tmp_path / "agent_trees"
    run = _chain(work, trees, "open_chain_0", "aa")
    exp = _experiment(run, 1, loop_rounds=1)
    old = time.time() - 5 * 3600
    for root, dirs, files in os.walk(run):
        for name in dirs + files:
            os.utime(os.path.join(root, name), (old, old))
    os.utime(run, (old, old))
    c = campaign_status(work)["chains"][0]
    assert c["state"] == "QUIET" and c["idle_hours"] > 4.9
    assert campaign_status(work, quiet_hours=6)["chains"][0]["state"] == "running"
    (exp / "loop" / "history.json").touch()  # a change: running again
    assert campaign_status(work)["chains"][0]["state"] == "running"


def test_a_smoke_cell_adds_no_participant_cost_and_a_missing_run_is_not_started(tmp_path):
    from src.rsa_open.status import campaign_status

    work, trees = tmp_path / "w", tmp_path / "agent_trees"
    run = _chain(work, trees, "open_chain_0_smoke", "aa")
    _experiment(run, 1, kept=15)
    cdir = work / "cells" / "fixed_chain_3"
    cdir.mkdir(parents=True)
    report = campaign_status(work)
    by = {c["cell"]: c for c in report["chains"]}
    assert by["open_chain_0_smoke"]["participant_usd"] is None and report["participant_usd"] == 0
    assert by["fixed_chain_3"]["state"] == "not started"
