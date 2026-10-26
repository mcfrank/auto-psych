"""Live collection for the open campaign (src/rsa_open/live.py, PLAN.md Phase 3b item 6): the page
with the live hooks, the study's terms from its paradigm, one study recruiting at a time,
campaign-wide participant ids, exclusions, monitoring, the PI's review of an open chain's first
paradigm, and the chain writing a live experiment's data as it writes a simulated one's."""

import json
import threading
import time
from pathlib import Path

import pytest

from tests.test_rsa_open_screens import paradigm_of, units_of

ROOT = Path(__file__).resolve().parents[1]


def frozen_experiment(root, kind="choice", **design):
    """``<root>/experiment1`` with a frozen paradigm of ``kind`` and its lists (one per person)."""
    from src.rsa_open.screens import trial_lists

    p = paradigm_of(kind, **design)
    lists = trial_lists(p, units_of(p), seed=0, n_lists=p.design.n_participants)
    frozen = Path(root) / "experiment1" / "design" / "frozen"
    frozen.mkdir(parents=True)
    (frozen / "paradigm.json").write_text(json.dumps(p.model_dump(mode="json")))
    (frozen / "lists.json").write_text(json.dumps(lists))
    (frozen / "record.json").write_text(json.dumps({"paradigm_sha256": p.sha256, "submitted_by": "designer"}))
    return p, lists, Path(root)


def passing(trial):
    e = trial["expect"]
    return {"choice": e["choice"]}


def failing(trial):
    return {"choice": next(o for o in range(len(trial["objects"])) if o != trial["expect"]["choice"])}


def records(lst, *, comprehension="right", catch_ok=True, answer=None):
    """One person's jsPsych records, as the page writes them (`src/rsa_open/page.html`)."""
    recs = [{"task": "welcome", "time_elapsed": 1000}, {"task": "instructions", "time_elapsed": 9000}]
    if comprehension == "right":
        recs += [{"task": "comprehension", "check": 0, "correct": True, "attempt": 1, "time_elapsed": 12000}]
    elif comprehension == "second":
        recs += [{"task": "comprehension", "check": 0, "correct": False, "attempt": 1},
                 {"task": "instructions"}, {"task": "comprehension", "check": 0, "correct": True, "attempt": 2}]
    else:
        recs += [{"task": "comprehension", "check": 0, "correct": False, "attempt": 1},
                 {"task": "instructions"}, {"task": "comprehension", "check": 0, "correct": False, "attempt": 2}]
    clock = 15000
    for k, t in enumerate(lst["trials"]):
        clock += 4000
        if t["phase"] == "test":
            response = answer(t, k) if answer else {"choice": t["display_order"][k % len(t["display_order"])]}
        else:
            response = passing(t) if catch_ok or t["phase"] == "practice" else failing(t)
        recs.append({"task": "rsa_open_trial", "phase": t["phase"], "unit": t["unit"], "trial_number": k,
                     "objects": t["objects"], "heard": t["heard"], "expect": t.get("expect"),
                     "display_order": t["display_order"], "response": response, "rt": 1500, "time_elapsed": clock,
                     "list_index": lst["list_index"], "paradigm_sha256": lst.get("paradigm_sha256")})
    return recs


def stored(pid, lst, recs, study="S1", at=0):
    return {"trials": [{"format": "rsa_jspsych_json", "data": json.dumps(recs)}], "list_index": lst["list_index"],
            "prolific_pid": pid, "participant_id_str": pid, "prolific_study_id": study,
            "submitted_at_client": f"2026-10-27T00:00:{at:02d}Z"}


def live_cfg(tmp_path, **kw):
    from src.rsa_open.live import load_live_config

    values = dict(prolific_mode="live", confirm_live_recruitment=True, firebase_project="auto-psych-2c5da",
                  run_label="open-c0", repo_root=tmp_path / "deploy", private_dir=tmp_path / "private",
                  campaign_dir=tmp_path / "campaign", max_wait_sec=0.3)
    values.update(kw)
    return load_live_config(**values)


# --------------------------------------------------------------------------- terms and config

def test_the_committed_campaign_config_is_valid_and_states_no_study_terms_of_its_own(tmp_path):
    import yaml

    from src.rsa_open.live import OPEN_LIVE_CONFIG, hosting_site, read_campaign_config

    cfg = read_campaign_config()
    assert cfg["confirm_live_recruitment"] is False  # set deliberately for a paid launch, never committed true
    assert cfg["prolific"]["exclude_earlier_participants_from"] == "project"
    assert hosting_site(cfg, "fixed", 3) == "auto-psych-2c5da-ro-fixed-c3"
    bad = yaml.safe_load(OPEN_LIVE_CONFIG.read_text())
    bad["prolific"]["reward"] = 50
    (tmp_path / "bad.yaml").write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match="each study's"):
        read_campaign_config(tmp_path / "bad.yaml")


def test_a_study_s_pay_minutes_and_text_are_its_paradigm_s_under_the_neutral_name(tmp_path):
    from src.rsa_open.fixed import SPEC_PATH
    from src.rsa_open.live import recruitment_overrides
    from src.rsa_open.paradigm import load_paradigm

    o = recruitment_overrides(load_paradigm(SPEC_PATH), live_cfg(tmp_path))
    assert o == {"reward": 50, "estimated_completion_time": 3, "name": "What do people mean?",
                 "description": o["description"]}
    assert "About 3 minutes" in o["description"] and "Bob" not in o["description"]
    p = paradigm_of("rating", minutes=4.5, pay_usd=0.80)
    assert recruitment_overrides(p, live_cfg(tmp_path))["estimated_completion_time"] == 5  # Prolific's whole minutes


def test_per_study_overrides_replace_the_rendered_settings(tmp_path, monkeypatch):
    import src.runtime.prolific as rp
    from src.pipelines.outer_loop.deployment.prolific import load_recruitment_config

    path = tmp_path / "prolific_config.yaml"
    path.write_text("reward_per_hour: 1200\nestimated_completion_time: 5\nname: closed\n")
    monkeypatch.setattr(rp, "prolific_config_path", lambda project: path)
    cfg = load_recruitment_config("rsa_reference", 10, {"reward": 80, "estimated_completion_time": 4, "name": "x"})
    assert cfg["reward"] == 80 and "reward_per_hour" not in cfg and cfg["estimated_completion_time"] == 4
    assert load_recruitment_config("rsa_reference", 10)["reward_per_hour"] == 1200
    with pytest.raises(ValueError, match="not among"):
        load_recruitment_config("rsa_reference", 10, {"total_available_places": 5})


def test_live_recruitment_is_double_gated(tmp_path):
    with pytest.raises(ValueError, match="confirm_live_recruitment"):
        live_cfg(tmp_path, confirm_live_recruitment=False)
    live_cfg(tmp_path, prolific_mode="test", confirm_live_recruitment=False)


# --------------------------------------------------------------------------- the page

def test_the_live_page_has_the_hooks_and_no_consent_screen_of_its_own(tmp_path):
    from src.pipelines.outer_loop.deployment.firebase import (
        CONSENT_GATE_MARKER, _has_working_submit, ensure_consent_gate)
    from src.rsa.live.site import LIVE_MARKER
    from src.rsa_open.live import build_site

    p, lists, root = frozen_experiment(tmp_path)
    html = build_site(lists, root / "experiment1").read_text()
    assert html.count(f'<script id="{LIVE_MARKER}">') == 1 and _has_working_submit(html)
    assert "const CONSENT_HTML = null" in html and "[CONSENT PLACEHOLDER]" not in html
    assert html.index(LIVE_MARKER) < html.index("const LISTS")  # the hooks exist before the page asks for them
    assert CONSENT_GATE_MARKER in ensure_consent_gate(html, "<p>consent</p>")


def test_the_live_page_takes_its_list_from_assign_and_posts_what_collection_reads(tmp_path):
    """In Chromium: the page asks /assign for its list and posts its data to /submit, and the
    collection reads that post back as the person's rows."""
    from tests.rsa_experiment_browser import CHROMIUM, chromium_available, fetch_cdn
    from tests.test_rsa_open_page import answer_everything

    if not chromium_available():
        pytest.skip(f"Playwright or Chromium ({CHROMIUM}) missing")
    try:
        cdn = fetch_cdn()
    except OSError as e:
        pytest.skip(f"jsPsych CDN unreachable: {e}")
    from playwright.sync_api import sync_playwright

    from src.rsa_open.live import build_site, exclusions, person

    p, lists, root = frozen_experiment(tmp_path, "rating", trials_per_participant=4, catch_trials=1)
    index = build_site(lists, root / "experiment1")
    site = "https://site.test/e1-open-c0/"
    posts, assigns, blocked = [], [], []

    def handle(route):
        url, req = route.request.url, route.request
        if url in cdn:
            ct, body = cdn[url]
            return route.fulfill(status=200, body=body, headers={"Content-Type": ct})
        if url.startswith(site) and url.split("?")[0] == site:
            return route.fulfill(status=200, body=index.read_text(), headers={"Content-Type": "text/html"})
        if url == site + "auto_psych_config.json":
            return route.fulfill(status=200, body=json.dumps({"collection_session_id": "sess", "prolific_study_id": "S1"}),
                                 headers={"Content-Type": "application/json"})
        if url == "https://site.test/assign":
            assigns.append(json.loads(req.post_data))
            return route.fulfill(status=200, body=json.dumps({"list_index": 7}), headers={"Content-Type": "application/json"})
        if url == "https://site.test/submit":
            posts.append(json.loads(req.post_data))
            return route.fulfill(status=200, body="{}", headers={"Content-Type": "application/json"})
        blocked.append(url)
        return route.abort()

    with sync_playwright() as pw:
        b = pw.chromium.launch(executable_path=str(CHROMIUM))
        page = b.new_page(viewport={"width": 1100, "height": 900})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.route("**/*", handle)
        page.goto(site + "?PROLIFIC_PID=abc123&STUDY_ID=S1&SESSION_ID=x")
        answer_everything(page)
        b.close()
    assert not blocked and not errors, (blocked, errors)
    assert assigns == [{"collection_session_id": "sess", "n_lists": len(lists["lists"]), "participant_key": "abc123"}]
    (post,) = posts
    assert post["list_index"] == 7 and post["prolific_pid"] == "abc123" and post["prolific_study_id"] == "S1"
    who = person("p0", post, lists)
    assert not who.problems and len(who.rows) == len(lists["lists"][7]["trials"])
    assert who.comprehension_final is True  # wrong at first, right on the second reading
    assert exclusions(who, repeat=False) == []  # the top rating passes the catch trial
    assert {r["response"]["rating"] for r in who.rows if r["phase"] == "test"} == {5}


# --------------------------------------------------------------------------- people

def test_exclusions_are_a_missed_catch_failed_checks_a_repeat_or_an_incomplete_record(tmp_path):
    from src.rsa_open.live import exclusions, person

    p, lists, _ = frozen_experiment(tmp_path)
    lst = lists["lists"][0]
    ok = person("p0", stored("a", lst, records(lst)), lists)
    assert ok.comprehension_final and not ok.problems and exclusions(ok, False) == []
    assert exclusions(person("p1", stored("b", lst, records(lst, comprehension="second")), lists), False) == []
    assert exclusions(person("p2", stored("c", lst, records(lst, comprehension="wrong")), lists), False) == [
        "failed the comprehension checks on the second reading"]
    assert exclusions(person("p3", stored("d", lst, records(lst, catch_ok=False)), lists), False) == [
        "missed a catch trial"]
    assert exclusions(ok, True) == ["took part in an earlier study of the campaign"]
    short = records(lst)[:-2]
    assert any("incomplete" in w for w in exclusions(person("p4", stored("e", lst, short), lists), False))
    other = person("p5", stored("f", lists["lists"][1], records(lists["lists"][1]), at=1) | {"list_index": 10**6}, lists)
    assert other.problems == [f"list {10**6} is not one of this design's"]


@pytest.mark.parametrize("kind,response,key", [
    ("choice", lambda t, k: {"choice": t["display_order"][0]}, "position 0"),
    ("rating", lambda t, k: {"probe": t.get("probe"), "rating": 4}, "rating 4"),
    ("slider", lambda t, k: {"probe": t.get("probe"), "value": 51 + k % 2}, "value ~50"),
    ("binary_judgment", lambda t, k: {"target": 0, "utterance": 0, "answer": True}, "answer True"),
])
def test_the_monitor_flags_the_same_answer_on_every_trial_for_each_response_type(kind, response, key):
    from src.rsa_open.live import answer_key, degenerate

    rows = [{"phase": "test", "display_order": [2, 0, 1], "response": response({"display_order": [2, 0, 1]}, k)}
            for k in range(8)]
    assert answer_key(rows[0]) == key
    assert degenerate(rows).startswith(key)
    varied = [dict(r, display_order=[k % 3, (k + 1) % 3, (k + 2) % 3]) for k, r in enumerate(rows)]
    if kind == "choice":
        assert degenerate(varied) is None  # the same object, a different screen position each time
    pair = {"phase": "test", "display_order": [0, 1], "response": {"choice": 1, "second": {"probe": 0, "rating": 2}}}
    assert answer_key(pair) == "position 1 | rating 2"


# --------------------------------------------------------------------------- the campaign

def test_campaign_ids_are_unique_and_a_person_seen_in_another_study_is_a_repeat(tmp_path):
    from src.rsa_open.live import campaign_ids

    ids, repeats = campaign_ids(tmp_path, "open/chain_0/experiment1/open-c0", ["a", "b"])
    assert ids == {"a": "p0", "b": "p1"} and repeats == []
    again, repeats = campaign_ids(tmp_path, "open/chain_0/experiment1/open-c0", ["a", "b"])
    assert again == ids and repeats == []  # a resumed collection reads its own people back
    ids, repeats = campaign_ids(tmp_path, "fixed/chain_1/experiment1/fixed-c1", ["c", "b"])
    assert ids == {"c": "p2", "b": "p1"} and repeats == ["p1"]


def test_one_study_recruits_at_a_time_across_the_campaign(tmp_path):
    from src.rsa_open.live import study_slot

    order = []

    def second():
        with study_slot(tmp_path, 1, poll_sec=0.05, label="second"):
            order.append("second in")

    with study_slot(tmp_path, 1, label="first") as k:
        assert k == 0
        t = threading.Thread(target=second)
        t.start()
        time.sleep(0.3)
        order.append("first out")
    t.join(5)
    assert order == ["first out", "second in"]
    with study_slot(tmp_path, 2, label="a"), study_slot(tmp_path, 2, poll_sec=0.05, label="b") as k:
        assert k == 1  # two slots: a second study need not wait


# --------------------------------------------------------------------------- one experiment, end to end

@pytest.fixture
def fake_services(monkeypatch):
    import src.pipelines.outer_loop.collect as main_collect
    import src.pipelines.outer_loop.deployment.local as local
    import src.rsa.live.collect as closed
    from src.pipelines.outer_loop.deployment.manifest import manifest_path

    calls = {"deploy": [], "paused": [], "responses": []}

    def deploy(*, exp_dir, prolific_mode, n_participants, run_label, recruitment_overrides, **kw):
        calls["deploy"].append(dict(mode=prolific_mode, n=n_participants, label=run_label,
                                    overrides=recruitment_overrides, project=kw["project_id"]))
        assert (Path(exp_dir) / "experiment" / "index.html").exists()
        path = manifest_path(exp_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"prolific_mode": prolific_mode, "prolific_study_id": "S1",
                                    "experiment_url": "https://x/e1/", "results_api_url": "https://x",
                                    "collection_session_id": "sess", "project_id": "rsa_reference",
                                    "experiment_id": "rsa_reference_experiment1", "run_id": 1,
                                    "deployment_id": "deploy_x", "study_id": "study_rsa_reference",
                                    "deploy_target": "firebase", "agent_backend": "opencode",
                                    "collection_owner": "auto-psych", "firebase_project": "p",
                                    "firebase_region": "us-central1"}))
        return path

    monkeypatch.setattr(local, "run_deployment", deploy)
    monkeypatch.setattr(main_collect, "_poll_prolific_until_target", lambda study, target, log, **kw: target - 1)
    monkeypatch.setattr(main_collect, "_pause_unfilled_study", lambda *a: calls["paused"].append(a))
    monkeypatch.setattr(closed, "fetch_responses", lambda manifest: calls["responses"])
    return calls


def test_a_live_experiment_is_deployed_collected_and_cleaned(tmp_path, fake_services):
    from src.rsa_open.live import campaign_ids, collect

    p, lists, run = frozen_experiment(tmp_path / "run")
    L = lists["lists"]
    campaign_ids(tmp_path / "campaign", "fixed/chain_1/experiment1/fixed-c1", ["seen-before"])
    fake_services["responses"][:] = [
        stored("a", L[0], records(L[0]), at=1), stored("b", L[1], records(L[1]), at=2),
        stored("a", L[0], records(L[0]), at=3),                                         # a reload's second post
        stored("c", L[2], records(L[2], catch_ok=False), at=4),
        stored("d", L[3], records(L[3], comprehension="wrong"), at=5),
        stored("seen-before", L[4], records(L[4]), at=6),
        stored("z", L[5], records(L[5]), study="DRAFT", at=7),                           # a preview of the draft
    ]
    cfg = live_cfg(tmp_path)
    rows, record, seconds = collect(exp_dir=run / "experiment1", chain="chain_0", arm="fixed", n=1, cfg=cfg)
    (d,) = fake_services["deploy"]
    assert d["mode"] == "live" and d["n"] == p.design.n_participants and d["project"] == "rsa_reference"
    assert d["overrides"]["reward"] == round(p.design.pay_usd * 100) and d["label"] == "open-c0"
    assert fake_services["paused"]  # one short of its target after the wait: paused, partial data kept
    assert {r["participant"] for r in rows} == {"p1", "p2"}  # a, b; "seen-before" took p0 in another study
    assert record["kept"] == 2 and record["excluded"] == 3 and record["second_submissions"] == 1
    assert record["other_study"] == 1 and record["exclusion_reasons"] == {
        "missed a catch trial": 1, "failed the comprehension checks on the second reading": 1,
        "took part in an earlier study of the campaign": 1}
    assert len(seconds) == 2 and all(s > 0 for s in seconds)
    raw = json.loads((tmp_path / "private" / "raw_collected" / "experiment1.json").read_text())
    assert {r["prolific_pid"] for r in raw} >= {"a", "b"}  # the private copy, with Prolific ids
    assert "abc" not in json.dumps(record) and not any(k in json.dumps(rows) for k in ('"a"', "seen-before"))
    assert (run / "experiment1" / "deployment" / "monitor.json").exists()
    # A job that stopped after the wait ended resumes at the fetch, without waiting or deploying again.
    import src.pipelines.outer_loop.collect as main_collect

    main_collect._poll_prolific_until_target = lambda *a, **k: pytest.fail("waited again")
    rows2, record2, _ = collect(exp_dir=run / "experiment1", chain="chain_0", arm="fixed", n=1, cfg=cfg)
    assert rows2 == rows and record2["recruited"] == record["recruited"] and len(fake_services["deploy"]) == 1


def test_an_open_chain_s_first_paradigm_is_published_only_once_the_pi_approves_it(tmp_path, fake_services):
    from src.rsa.live.collect import DraftOnly
    from src.rsa_open.live import AwaitingReview, collect
    from src.rsa_open.review import write_review

    p, lists, run = frozen_experiment(tmp_path / "run")
    exp = run / "experiment1"
    with pytest.raises(AwaitingReview, match="has not been reviewed"):
        collect(exp_dir=exp, chain="chain_0", arm="open", n=1, cfg=live_cfg(tmp_path))
    assert fake_services["deploy"] == []  # nothing deployed or published
    with pytest.raises(DraftOnly):  # a test launch needs no approval: it is how the PI sees the page
        collect(exp_dir=exp, chain="chain_0", arm="open", n=1,
                cfg=live_cfg(tmp_path, prolific_mode="test", confirm_live_recruitment=False))
    (exp / "deployment").rename(exp / "deployment.test")
    write_review(exp / "design" / "frozen", "mcfrank", reject="the checks are ambiguous")
    with pytest.raises(AwaitingReview, match="not approved"):
        collect(exp_dir=exp, chain="chain_0", arm="open", n=1, cfg=live_cfg(tmp_path))
    (exp / "design" / "frozen" / "review.json").unlink()
    write_review(exp / "design" / "frozen", "mcfrank")
    L = lists["lists"]
    fake_services["responses"][:] = [stored("a", L[0], records(L[0]))]
    rows, record, _ = collect(exp_dir=exp, chain="chain_0", arm="open", n=1, cfg=live_cfg(tmp_path))
    assert record["kept"] == 1


def test_a_review_names_the_paradigm_it_approves(tmp_path):
    from src.rsa_open.live import AwaitingReview, check_review
    from src.rsa_open.review import describe, write_review

    p, lists, run = frozen_experiment(tmp_path)
    frozen = run / "experiment1" / "design" / "frozen"
    assert p.name in describe(frozen)
    write_review(frozen, "mcfrank")
    assert check_review(frozen, p.sha256)["by"] == "mcfrank"
    with pytest.raises(AwaitingReview, match="another paradigm"):
        check_review(frozen, "0" * 64)
    with pytest.raises(FileExistsError):
        write_review(frozen, "mcfrank")


def test_a_smoke_test_recruits_its_own_count_and_the_chain_stops_after_collecting(tmp_path):
    from src.rsa_open.chain import Chain, ChainConfig

    with pytest.raises(ValueError, match="stop_after_collect"):
        Chain(ChainConfig(chain="chain_0", seeds_dir=tmp_path, literature=None, out_dir=tmp_path, ground_truth=None,
                          live=live_cfg(tmp_path, participants=15)), designer_spawn=None, theorist_spawn=None)


def test_the_chain_writes_a_live_experiment_s_data_as_a_simulated_one_s(tmp_path, fake_services):
    from src.rsa_open.chain import Chain, ChainConfig
    from src.rsa_open.corpus import Block
    from src.rsa_open.loop.fitting import read_corpus

    p, lists, run = frozen_experiment(tmp_path / "run")
    L = lists["lists"]
    fake_services["responses"][:] = [stored(f"x{k}", L[k], records(L[k]), at=k) for k in range(5)]
    chain = Chain(ChainConfig(chain="chain_0", arm="fixed", seeds_dir=tmp_path, literature=None, out_dir=run,
                              ground_truth=None, live=live_cfg(tmp_path, participants=5), stop_after_collect=True),
                  designer_spawn=None, theorist_spawn=None)
    out = chain.collect(1)
    data = run / "experiment1" / "data"
    (block,) = read_corpus(out)
    assert isinstance(block, Block) and len(block.responses) == 5 * p.design.trials_per_participant
    assert json.loads((data / "participants.json").read_text())["smoke_test"] is True
    assert json.loads((data / "timing.json").read_text())[0]["n_people"] == 5
    assert fake_services["deploy"][0]["n"] == 5  # the smoke test's count, not the paradigm's N
    assert chain.collect(1) == out  # collected once
    with pytest.raises(RuntimeError, match="deployment but no frozen design"):
        (run / "experiment1" / "design" / "frozen" / "record.json").unlink()
        chain.design(1)


def test_a_job_does_not_start_a_stage_it_cannot_finish(tmp_path):
    from src.rsa_open.chain import Chain, ChainConfig, OutOfJobTime

    calls = []
    chain = Chain(ChainConfig(chain="chain_0", seeds_dir=tmp_path, literature=None, out_dir=tmp_path / "run",
                              ground_truth=None, deadline=time.time() + 3 * 3600),
                  designer_spawn=None, theorist_spawn=None)
    chain.design = lambda n: calls.append(("design", n))
    with pytest.raises(OutOfJobTime, match="3.0 h of the job left, under the 8 h a design needs"):
        chain.run()
    assert calls == []
    frozen = tmp_path / "run" / "experiment1" / "design" / "frozen"
    frozen.mkdir(parents=True)
    (frozen / "record.json").write_text("{}")  # designed already: collection may start
    chain.collect = lambda n: calls.append(("collect", n))
    with pytest.raises(OutOfJobTime, match="a loop needs"):
        chain.run()
    assert calls == [("design", 1), ("collect", 1)]


# --------------------------------------------------------------------------- the job and its launcher

SLURM = ROOT / "scripts" / "rsa" / "slurm"
LAUNCH = ROOT / "scripts" / "rsa" / "live" / "open_launch.sh"


def _launch(tmp_path, env, stdin=""):
    """open_launch.sh against stand-ins for Sherlock's modules and sbatch (which records its calls)."""
    import os
    import subprocess

    bin_ = tmp_path / "bin"
    bin_.mkdir(exist_ok=True)
    for tool in ("ml", "bwrap", "node", "npm"):
        (bin_ / tool).write_text("#!/bin/sh\nexit 0\n")
        (bin_ / tool).chmod(0o755)
    calls = tmp_path / "calls"
    (bin_ / "sbatch").write_text(f'#!/bin/sh\necho "$* | CHAIN=$CHAIN ARM=$ARM CONFIRM=$CONFIRM_LIVE_RECRUITMENT '
                                 f'PARTICIPANTS=$PARTICIPANTS" >> {calls}\necho $$\n')
    (bin_ / "sbatch").chmod(0o755)
    venv = tmp_path / "scratch" / "auto-psych" / "rsa_open_live" / "venv"
    venv.parent.mkdir(parents=True, exist_ok=True)
    if not venv.exists():
        venv.symlink_to(ROOT / ".venv")
    full = dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}", REPO=str(ROOT), SCRATCH=str(tmp_path / "scratch"))
    full.update(env)
    out = subprocess.run(["bash", str(LAUNCH)], capture_output=True, text=True, env=full, input=stdin)
    return out, (calls.read_text().splitlines() if calls.exists() else [])


@pytest.mark.skipif(not (ROOT / ".venv").exists(), reason="needs the project's venv")
def test_each_chain_gets_a_ladder_of_jobs_and_a_stopped_chain_is_not_relaunched(tmp_path):
    out, calls = _launch(tmp_path, dict(PROLIFIC_MODE="test", ARM="fixed", CHAINS="0 1", LINKS="3"))
    assert out.returncode == 0, out.stderr
    assert "400 people x $0.50 for 3 min: $266.67 a study" in out.stderr
    assert len(calls) == 6 and all("--partition=normal --time=48:00:00" in c for c in calls)
    assert [("--dependency=afterany:" in c) for c in calls] == [False, True, True] * 2
    assert all("CONFIRM= " in c for c in calls)  # a test launch is never confirmed
    stop = tmp_path / "scratch" / "auto-psych" / "rsa_open_live" / "cells" / "fixed_chain_0" / "STOP"
    stop.parent.mkdir(parents=True)
    stop.write_text("then\tawaiting review\n")
    out, calls2 = _launch(tmp_path, dict(PROLIFIC_MODE="test", ARM="fixed", CHAINS="0", LINKS="1"))
    assert "relaunch with CLEAR_STOP=1" in out.stderr and calls2 == calls
    out, calls3 = _launch(tmp_path, dict(PROLIFIC_MODE="test", ARM="fixed", CHAINS="0", LINKS="1", CLEAR_STOP="1"))
    assert len(calls3) == 7 and not stop.exists() and list(stop.parent.glob("STOP.*"))


@pytest.mark.skipif(not (ROOT / ".venv").exists(), reason="needs the project's venv")
def test_a_live_launch_needs_the_frozen_tag_and_a_typed_yes(tmp_path):
    out, calls = _launch(tmp_path, dict(PROLIFIC_MODE="live", ARM="open", CHAINS="0"), stdin="yes\n")
    assert out.returncode != 0 and "frozen tag" in out.stderr and calls == []
    out, calls = _launch(tmp_path, dict(PROLIFIC_MODE="live", ARM="open", CHAINS="0", PARTICIPANTS="15",
                                        SMOKE_FROM="/x/experiment1", LINKS="1"), stdin="no\n")
    assert out.returncode != 0 and "not launched" in out.stderr and calls == []
    out, calls = _launch(tmp_path, dict(PROLIFIC_MODE="live", ARM="open", CHAINS="0", PARTICIPANTS="15",
                                        SMOKE_FROM="/x/experiment1", LINKS="1"), stdin="yes\n")
    assert out.returncode == 0, out.stderr
    (call,) = calls
    assert "CONFIRM=1" in call and "PARTICIPANTS=15" in call and "rsa_open_open_c0_smoke" in call


def test_the_live_job_is_double_gated_stops_on_record_and_resumes_after_a_time_limit():
    text = (SLURM / "open_live.sbatch").read_text()
    assert 'confirm_live_recruitment)" == "True"' in text and 'CONFIRM_LIVE_RECRUITMENT:-}" == "1"' in text
    for v in ("PROLIFIC_API_TOKEN", "AUTO_PSYCH_RESULTS_TOKEN", "FIREBASE_TOKEN"):
        assert v in text
    assert '--private-dir "$CELL_DIR/private"' in text and '--campaign-dir "$CAMPAIGN_DIR"' in text
    assert 'CAMPAIGN_DIR="$WORK_ROOT/private/campaign"' in text  # outside every agents' tree
    assert "a live chain runs on a frozen tag" in text and 'git rev-parse "$CODE_TAG^{commit}"' in text
    assert 'if [[ -f "$CELL_DIR/STOP" ]]' in text and "--job-deadline" in text
    for rc in ("0)", "3)", "4)", "5)", "*)"):
        assert f"  {rc}" in text
    five = text.split("  5)")[1].split(";;")[0]
    assert "stop " not in five  # out of job time: no STOP, the next job resumes
    from src.rsa_open.run_chain import EXIT_AWAITING_REVIEW, EXIT_DRAFT_ONLY, EXIT_OUT_OF_JOB_TIME

    assert (EXIT_AWAITING_REVIEW, EXIT_DRAFT_ONLY, EXIT_OUT_OF_JOB_TIME) == (3, 4, 5)
