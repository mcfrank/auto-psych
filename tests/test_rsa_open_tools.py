"""The designer's tools (src/rsa_open/tools.py): commands, reports, the call budget, submit."""

import json
import shutil

import numpy as np
import pytest

from tests.test_rsa_open_design import SEEDS, response_block, small_paradigm


@pytest.fixture(scope="module")
def base_workspace(tmp_path_factory):
    from src.rsa.model_file import prior_draws
    from src.rsa_open.design import write_context
    from src.rsa_open.fit import TheoryModel
    from src.rsa_open.port import port

    root = tmp_path_factory.mktemp("ws")
    src = root / "src"
    src.mkdir()
    theories = {}
    for name in ("literal_listener", "rsa_l1"):
        path = src / f"{name}.py"
        path.write_text(port((SEEDS / f"{name}.py").read_text(), name=f"{name}.py"))
        draws = prior_draws(TheoryModel(path).params, 40, 0)
        theories[name] = (path, {k: np.asarray([d[k] for d in draws]) for k in draws[0]})
    ws = root / "ws"
    write_context(ws / "context", theories)
    (ws / "inputs").mkdir()
    from src.rsa_open.timing import CLOSED_CAMPAIGN, write_records

    write_records(ws / "inputs" / "timing.json", CLOSED_CAMPAIGN)
    (ws / "work").mkdir()
    return ws


@pytest.fixture
def ws(base_workspace, tmp_path):
    from src.rsa_open.tools import Workspace

    root = tmp_path / "ws"
    shutil.copytree(base_workspace, root)
    return Workspace(root)


def write_spec(ws, response=None, **design):
    spec = small_paradigm(response or response_block("choice"))
    data = spec.model_dump(mode="json")
    data["design"].update(design)
    path = ws.work / f"p{len(list(ws.work.glob('*.json')))}.json"
    path.write_text(json.dumps(data))
    return path


def test_validate_reports_a_valid_spec_and_logs_the_call(ws, capsys):
    from src.rsa_open.tools import main

    spec = write_spec(ws)
    assert main(["validate", "--workspace", str(ws.root), "--spec", str(spec)]) == 0
    out = capsys.readouterr().out
    assert "valid" in out and "39 tool calls left" in out
    assert "time: estimated" in out and "9 choice experiment(s)" in out
    assert [c["command"] for c in ws.calls()] == ["validate"]


def test_an_invalid_spec_lists_its_problems_and_still_costs_a_call(ws, capsys):
    from src.rsa_open.tools import main

    spec = write_spec(ws, n_participants=10)
    assert main(["validate", "--workspace", str(ws.root), "--spec", str(spec)]) == 1
    assert "n_participants" in capsys.readouterr().out
    assert ws.calls()[0]["outcome"].startswith("refused")


def test_the_budget_refuses_the_call_past_it(ws, capsys):
    from src.rsa_open.tools import MAX_TOOL_CALLS, main

    spec = write_spec(ws)
    ws.log.write_text("".join(json.dumps({"command": "validate"}) + "\n" for _ in range(MAX_TOOL_CALLS)))
    assert main(["validate", "--workspace", str(ws.root), "--spec", str(spec)]) == 1
    assert "budget" in capsys.readouterr().out
    assert len(ws.calls()) == MAX_TOOL_CALLS  # a refused-for-budget call is not run, nor logged


def test_a_directory_that_is_not_a_workspace_is_refused(tmp_path, capsys):
    from src.rsa_open.tools import main

    assert main(["validate", "--workspace", str(tmp_path), "--spec", str(tmp_path / "x.json")]) == 1
    assert "not a designer workspace" in capsys.readouterr().out


def test_eig_writes_a_report_named_by_the_spec_and_scores_against_the_floor(ws, monkeypatch, capsys):
    import src.rsa_open.design as design
    from src.rsa_open.tools import main

    calls = []

    def fake_reference(ctx, *, n_draws, n_scenarios, seed):
        calls.append(seed)
        return {"bits_per_dollar": 1e-9}

    monkeypatch.setattr(design, "reference_report", fake_reference)
    monkeypatch.setattr(design, "DEFAULT_DRAWS", 16)
    monkeypatch.setattr(design, "DEFAULT_SCENARIOS", 100)
    spec = write_spec(ws)
    assert main(["eig", "--workspace", str(ws.root), "--spec", str(spec)]) == 0
    out = capsys.readouterr().out
    assert "MEETS" in out
    (report_path,) = ws.reports.glob("eig_*.json")
    report = json.loads(report_path.read_text())
    assert report_path.name == f"eig_{report['paradigm_sha256'][:12]}.json"
    assert report["meets_floor"] and calls == [0]  # the reference's seed is the context's, not the spec's


def test_the_eig_seed_is_fixed_by_the_spec():
    from src.rsa_open.tools import seed_for

    assert seed_for("a" * 64, "eig") == seed_for("a" * 64, "eig") != seed_for("b" * 64, "eig")
    assert seed_for("a" * 64, "eig") != seed_for("a" * 64, "link_check")


def _report(ws, command, sha, **fields):
    ws.reports.mkdir(exist_ok=True)
    ws.report_path(command, sha).write_text(json.dumps({"paradigm_sha256": sha, **fields}))


def test_submit_needs_passing_reports_of_this_exact_spec(ws, capsys):
    from src.rsa_open.paradigm import load_paradigm
    from src.rsa_open.tools import main

    spec = write_spec(ws)
    sha = load_paradigm(spec).sha256
    argv = ["submit", "--workspace", str(ws.root), "--spec", str(spec)]
    assert main(argv) == 1
    out = capsys.readouterr().out
    assert "no eig report" in out and "no link-check report" in out
    _report(ws, "eig", sha, meets_floor=False)
    _report(ws, "link-check", sha, converged=True, recovered=False, misses={"lapse": 2})
    assert main(argv) == 1
    out = capsys.readouterr().out
    assert "below the floor" in out and "link check failed" in out
    assert not ws.submission.exists()
    _report(ws, "eig", sha, meets_floor=True)
    _report(ws, "link-check", sha, converged=True, recovered=True, misses={})
    assert main(argv) == 0
    assert json.loads(ws.submission.read_text())["sha256"] == sha
    # Reports of another spec do not count.
    other = write_spec(ws, n_participants=130)
    assert main(["submit", "--workspace", str(ws.root), "--spec", str(other)]) == 1


@pytest.mark.slow  # three NUTS fits
def test_link_check_runs_the_recovery_test_in_the_paradigm(ws):
    from src.rsa_open.tools import LinkCheck, run

    spec = write_spec(ws, response_block("rating"),
                      n_participants=140, trials_per_participant=10)
    report = run(LinkCheck(workspace=ws.root, spec=spec, theory="rsa_l1"))
    assert report["n_trials"] == 1400 and report["theory"] == "rsa_l1"
    assert report["converged"] and report["recovered"], report["misses"]
    assert ws.report_path("link-check", report["paradigm_sha256"]).exists()


def test_validate_and_eig_report_the_pilot_s_time_estimate(ws, monkeypatch):
    import src.rsa_open.design as design
    from src.rsa_open.pilot import TIME_TOLERANCE
    from src.rsa_open.tools import Eig, Validate, run

    monkeypatch.setattr(design, "reference_report", lambda ctx, **kw: {"bits_per_dollar": 1e-9})
    monkeypatch.setattr(design, "DEFAULT_DRAWS", 16)
    monkeypatch.setattr(design, "DEFAULT_SCENARIOS", 100)
    spec = write_spec(ws)
    for command in (Validate, Eig):
        t = run(command(workspace=ws.root, spec=spec))["time"]
        assert t["declared_minutes"] == 4 and t["estimated_minutes"] > 0
        assert t["within_tolerance"] == (abs(t["estimated_minutes"] / 4 - 1) <= TIME_TOLERANCE)


def test_a_workspace_without_timing_records_is_refused(ws, capsys):
    from src.rsa_open.tools import main

    ws.timing.unlink()
    spec = write_spec(ws)
    assert main(["validate", "--workspace", str(ws.root), "--spec", str(spec)]) == 1
    assert "timing.json" in capsys.readouterr().out


def test_the_link_check_simulates_from_the_theory_with_the_most_parameters(ws):
    from src.rsa_open.design import load_context
    from src.rsa_open.tools import default_theory

    ctx = load_context(ws.context)
    assert len(ctx.theories["literal_listener"].params) == 0
    assert default_theory(ctx) == "rsa_l1"  # never the literal listener, which cannot move a temperature
