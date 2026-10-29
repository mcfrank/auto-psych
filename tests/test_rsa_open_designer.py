"""One designer cycle (src/rsa_open/designer.py): brief, submissions, the harness's checks, the
pilot, the retry and the frozen paradigm. The agent is a fake that runs the tools in-process;
the pilot's participants are scripted; EIG runs small and the link check is canned."""

import json
import math

import numpy as np
import pytest

from tests.test_rsa_open_design import SEEDS
from tests.test_rsa_open_pilot import Scripted
from tests.test_rsa_open_screens import paradigm_of


@pytest.fixture(scope="module")
def theories(tmp_path_factory):
    from src.rsa.model_file import prior_draws
    from src.rsa_open.fit import TheoryModel
    from src.rsa_open.port import port

    root = tmp_path_factory.mktemp("theories")
    out = {}
    for name in ("literal_listener", "rsa_l1"):
        path = root / f"{name}.py"
        path.write_text(port((SEEDS / f"{name}.py").read_text(), name=f"{name}.py"))
        draws = prior_draws(TheoryModel(path).params, 40, 0)
        out[name] = (path, {k: np.asarray([d[k] for d in draws]) for k in draws[0]})
    return out


HYPOTHESES = {"literal_listener": "Listeners pick uniformly among objects the word is true of.",
              "rsa_l1": "Listeners reason about a speaker who chooses informative words."}


@pytest.fixture
def fast(monkeypatch):
    """Small EIG settings, a fixed floor, and a canned link check (its real test is elsewhere)."""
    import src.rsa_open.design as design
    import src.rsa_open.tools as tools

    state = {"floor": 1e-9, "recovered": True}
    monkeypatch.setattr(design, "DEFAULT_DRAWS", 16)
    monkeypatch.setattr(design, "DEFAULT_SCENARIOS", 100)
    monkeypatch.setattr(design, "reference_report", lambda ctx, **kw: {"bits_per_dollar": state["floor"] / 0.5})

    def canned(ws, spec, **kw):
        from src.rsa_open.paradigm import load_paradigm

        return {"paradigm_sha256": load_paradigm(spec).sha256, "link": "choice", "theory": "rsa_l1",
                "n_trials": 1, "converged": True, "recovered": state["recovered"], "misses": {}}

    monkeypatch.setattr(tools, "link_check_report", canned)
    return state


def spec_for(minutes=None, response=None):
    """A choice paradigm (or one with ``response``) whose declared minutes are the pilot's estimate,
    rounded up (unless ``minutes``), paid $10 an hour: the pay floor holds at the estimate too."""
    from src.rsa_open.paradigm import parse_paradigm
    from src.rsa_open.pilot import estimated_minutes
    from src.rsa_open.screens import trial_lists
    from tests.test_rsa_open_paradigm import varied
    from tests.test_rsa_open_screens import units_of

    spec = varied(paradigm_of("choice").model_dump(mode="json"))
    if response is not None:
        spec["response"] = response
    if minutes is None:
        p = parse_paradigm(spec)
        minutes = math.ceil(estimated_minutes(trial_lists(p, units_of(p), seed=0, n_lists=1)) * 10) / 10
    spec["design"].update(minutes=minutes, pay_usd=math.ceil(minutes * 10 / 60 * 100) / 100)  # $10 an hour
    return spec


class FakeDesigner:
    """Each session writes the next spec of ``specs`` and runs the tools on it, as the agent would."""

    def __init__(self, specs, submit=True, score=True):
        self.specs, self.submit, self.score, self.prompts, self.sessions = list(specs), submit, score, [], 0
        self.refusals = []

    def __call__(self, ws, prompt, timeout_sec, log):
        from src.rsa_open.tools import Eig, LinkCheck, Submit, ToolRefused, Validate, run

        self.prompts.append(prompt)
        log.write_text("{}")
        spec = self.specs[min(self.sessions, len(self.specs) - 1)]
        self.sessions += 1
        path = ws.work / f"spec_{self.sessions}.json"
        path.write_text(json.dumps(spec))
        commands = (Validate,) + ((Eig, LinkCheck) if self.score else ()) + ((Submit,) if self.submit else ())
        for command in commands:
            try:
                run(command(workspace=ws.root, spec=path))
            except ToolRefused as err:  # the agent reads the refusal and ends its session
                self.refusals.append(str(err))
        (ws.work / "rationale.md").write_text("Why: a test.")
        return True


def scripted(lists_holder):
    return lambda i: Scripted(lists_holder["lists"], i)


def run(tmp_path, theories, spawn, **kw):
    from src.rsa_open.designer import run_designer

    return run_designer(tmp_path / "exp1", chain="c1", experiment=1, theories=theories, hypotheses=HYPOTHESES,
                        history=[], spawn=spawn, pilot_participants=4, code_commit="abc123", **kw)


@pytest.fixture
def pilot_lists(monkeypatch):
    """Scripted pilot participants for whatever lists the harness builds."""
    import src.rsa_open.pilot as pilot

    real = pilot.run_pilot

    def run_pilot(paradigm, lists, out, *, n_participants, participant_factory=None, max_workers=6, timing=None):
        return real(paradigm, lists, out, n_participants=n_participants,
                    participant_factory=lambda i: Scripted(lists, i, paradigm=paradigm), max_workers=1, timing=timing)

    monkeypatch.setattr(pilot, "run_pilot", run_pilot)


def test_a_passing_submission_is_checked_by_the_harness_and_frozen(tmp_path, theories, fast, pilot_lists):
    designer = FakeDesigner([spec_for()])
    record = run(tmp_path, theories, designer)
    frozen = tmp_path / "exp1" / "frozen"
    assert record["review_required"] and record["chain"] == "c1" and record["code_commit"] == "abc123"
    for name in ("paradigm.json", "design.json", "lists.json", "page_preview.html", "link_check.json",
                 "pilot/pilot.json", "record.json"):
        assert (frozen / name).exists(), name
    design = json.loads((frozen / "design.json").read_text())
    harness_seed_report = json.loads((tmp_path / "exp1" / "cycle_1" / "checks" / "1" / "eig.json").read_text())
    assert design["eig"] == harness_seed_report  # the design that runs is the harness's re-score
    assert design["eig"]["settings"]["seed"] != json.loads(
        next((tmp_path / "exp1" / "cycle_1" / "work" / "reports").glob("eig_*.json")).read_text())["settings"]["seed"]
    lists = json.loads((frozen / "lists.json").read_text())
    assert len(lists["lists"]) == 120 and lists["paradigm_sha256"] == record["paradigm_sha256"]
    assert designer.sessions == 1


def test_the_brief_states_the_charter_the_theories_the_tools_and_the_bounds(tmp_path, theories, fast, pilot_lists):
    designer = FakeDesigner([spec_for()])
    run(tmp_path, theories, designer)
    prompt = designer.prompts[0]
    for text in ("Grice (1975)", "rsa_l1", HYPOTHESES["rsa_l1"], "-m src.rsa_open.tools", "link-check",
                 "$266.67", "first experiment", "rationale.md", "no internet", "floor", "timing.md",
                 "--theory", "do not\nadd your own"):
        assert text in prompt, text


def test_a_failed_pilot_goes_back_to_the_designer_with_the_report(tmp_path, theories, fast, pilot_lists):
    designer = FakeDesigner([spec_for(minutes=4.5), spec_for()])
    record = run(tmp_path, theories, designer)
    assert designer.sessions == 2 and record["experiment"] == 1
    assert "did not pass" in designer.prompts[1] and "pilot: time" in designer.prompts[1]
    assert "2 submission(s) left" in designer.prompts[1]


def test_resubmitting_a_failed_spec_is_refused_without_checking_it_again(tmp_path, theories, fast, pilot_lists):
    designer = FakeDesigner([spec_for(minutes=4.5), spec_for(minutes=4.5), spec_for()])
    run(tmp_path, theories, designer)
    assert "already failed" in designer.prompts[2]
    assert not (tmp_path / "exp1" / "cycle_1" / "checks" / "2").exists()


def test_below_the_floor_on_the_harness_re_score_fails_the_submission(tmp_path, theories, fast, pilot_lists,
                                                                      monkeypatch):
    import src.rsa_open.designer as designer_mod
    from src.rsa_open.designer import DesignerFailed

    real = designer_mod.check_submission

    def raise_floor(ws, submitted, out, **kw):
        fast["floor"] = 1e9  # the designer's own eig met the floor; the harness's does not
        return real(ws, submitted, out, **kw)

    monkeypatch.setattr(designer_mod, "check_submission", raise_floor)
    designer = FakeDesigner([spec_for()], submit=True)
    with pytest.raises(DesignerFailed, match="without a submission"):
        run(tmp_path, theories, designer)
    assert "below the floor on the harness's re-score" in designer.prompts[1]
    assert any("below the floor" in r for r in designer.refusals)  # the agent's own submit now refuses it


def test_no_submission_fails_the_cycle_and_the_retry_hears_why_then_the_chain_stops(tmp_path, theories, fast,
                                                                                   pilot_lists):
    from src.rsa_open.designer import DesignerFailed

    designer = FakeDesigner([spec_for()], submit=False, score=False)  # nothing for the harness to submit either
    with pytest.raises(DesignerFailed, match="both designer cycles failed") as err:
        run(tmp_path, theories, designer)
    assert designer.sessions == 2  # one session per cycle: each ended without a submission
    assert "previous designer cycle" in designer.prompts[1] and "without a submission" in designer.prompts[1]
    assert "no spec of its had a full eig meeting the floor" in str(err.value)
    assert "cycle 2" in str(err.value)
    assert (tmp_path / "exp1" / "cycle_1" / "failure.txt").exists()


def test_a_cycle_stops_when_the_designer_s_time_is_spent(tmp_path, theories, fast, pilot_lists):
    from src.rsa_open.designer import DesignerFailed

    designer = FakeDesigner([spec_for(minutes=4.5)])
    with pytest.raises(DesignerFailed, match="time for the cycle was spent"):
        run(tmp_path, theories, designer, cycle_seconds=10)


def test_a_workspace_is_laid_out_once_with_its_inputs(tmp_path, theories, fast):
    from src.rsa_open.designer import EarlierParadigm, prepare_workspace
    from src.rsa_open.paradigm import parse_paradigm

    earlier = tmp_path / "earlier.json"
    earlier.write_text(json.dumps(spec_for()))
    from src.rsa_open.timing import CLOSED_CAMPAIGN, read_records

    ws = prepare_workspace(tmp_path / "ws", theories=theories,
                           history=[EarlierParadigm("forced choice", earlier, "| theory | elpd |\n")],
                           timing=CLOSED_CAMPAIGN)
    assert read_records(ws.timing) == list(CLOSED_CAMPAIGN)
    assert "closed campaign, chain 2, experiment 3" in (ws.inputs / "timing.md").read_text()
    assert (ws.inputs / "history" / "1_paradigm.json").exists() and (ws.inputs / "history" / "1_fits.md").exists()
    assert "ordered_beta" in (ws.inputs / "links.md").read_text()
    parse_paradigm(json.loads((ws.inputs / "format_example.json").read_text()))  # the example is valid
    with pytest.raises(FileExistsError):
        prepare_workspace(tmp_path / "ws", theories=theories, history=[], timing=CLOSED_CAMPAIGN)


def test_a_session_cut_off_by_its_time_limit_is_reported_as_such(tmp_path, theories, fast, pilot_lists, monkeypatch):
    import time

    import src.rsa_open.designer as designer_mod
    from src.rsa_open.designer import DesignerFailed

    monkeypatch.setattr(designer_mod, "MIN_SESSION_SECONDS", 0)

    def slow(ws, prompt, timeout_sec, log):
        time.sleep(timeout_sec + 0.1)  # works until the harness's limit, never submits
        return False

    with pytest.raises(DesignerFailed, match="reached its time limit"):
        run(tmp_path, theories, slow, cycle_seconds=1)


def test_a_session_that_ends_without_submitting_is_submitted_for_with_its_best_scored_spec(tmp_path, theories, fast,
                                                                                          pilot_lists):
    """Run B's experiment-3 designer had a full eig above the floor and ran out its time unsubmitted."""
    designer = FakeDesigner([spec_for()], submit=False)  # validates, scores and link-checks, never submits
    record = run(tmp_path, theories, designer)
    assert designer.sessions == 1 and record["submitted_by"] == "harness"
    submitted = json.loads((tmp_path / "exp1" / "cycle_1" / "submissions" / "1.json").read_text())
    assert submitted["submitted_by"] == "harness" and "ended without a submission" in submitted["why"]
    assert (tmp_path / "exp1" / "cycle_1" / "checks" / "1" / "eig.json").exists()  # checked like any submission


def test_a_designer_s_own_submission_is_recorded_as_its_own(tmp_path, theories, fast, pilot_lists):
    assert run(tmp_path, theories, FakeDesigner([spec_for()]))["submitted_by"] == "designer"


def test_the_brief_and_every_tool_give_the_session_s_deadline(tmp_path, theories, fast, pilot_lists):
    import re

    designer = FakeDesigner([spec_for()])
    run(tmp_path, theories, designer)
    assert re.search(r"this session ends at \*\*\d\d:\d\d", designer.prompts[0])
    assert "the harness submits your spec with the best full `eig`" in designer.prompts[0]
    deadline = json.loads((tmp_path / "exp1" / "cycle_1" / "inputs" / "deadline.json").read_text())
    assert deadline["clock"] in designer.prompts[0]
