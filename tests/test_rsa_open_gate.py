"""The feasibility gate (src/rsa_open/gate.py, PREREG_DRAFT.md §4): every item on a run tree made
to pass, and each one failing on its own."""

import json

import numpy as np
import pytest


def history(exp_n, best="a", trusted=("a", "b"), admitted=("c",)):
    block = f"exp{exp_n}"
    standing = {n: {"trusted": n in trusted, "by_block": {block: {"behind": 0.0 if n == best else 3.0, "dse": 2.0}}}
                for n in ("a", "b")}
    return [{"best_model": best, "standing": standing, "events": [{"outcome": "seeded", "name": "a"}]},
            {"best_model": best, "standing": standing,
             "events": [{"outcome": "admitted", "name": n} for n in admitted]}]


def write_experiment(run, n, *, ahead=60.0, exclusion_rate=0.1, same=0.0, admitted=("c",), trusted=("a", "b")):
    exp = run / f"experiment{n}"
    for d in ("design/frozen", "data", "loop"):
        (exp / d).mkdir(parents=True, exist_ok=True)
    (exp / "design" / "frozen" / "record.json").write_text(json.dumps(
        {"paradigm_sha256": "f" * 64, "submitted_by": "designer", "joint_eig_bits": 2.0}))
    (exp / "data" / "participants.json").write_text(json.dumps(
        {"collection": "live", "smoke_test": False, "comprehension_final": 0.9, "catch_pass": 0.95,
         "exclusion_rate": exclusion_rate, "same_answer": {"n": 0, "share": same}, "recruited": 200, "kept": 180}))
    (exp / "data" / "block.json").write_text("{}")
    (exp / "loop" / "history.json").write_text(json.dumps(history(n, admitted=admitted, trusted=trusted)))
    (exp / "loop" / "export.json").write_text(json.dumps({"best_model": "a", "live": ["a", "b"]}))
    rng = np.random.default_rng(n)
    units = np.repeat(np.arange(40), 5)
    chance = np.full(200, -1.1) + rng.normal(0, 0.05, 200)
    best = chance + ahead / 200 + rng.normal(0, 0.05, 200)
    (exp / "loop" / "newest_block_cv.json").write_text(json.dumps(
        {"block": f"exp{n}", "units": units.tolist(), "theories": {"a": best.tolist(), "b": chance.tolist()},
         "converged": {"a": True, "b": True}, "chance": chance.tolist(), "chance_converged": True,
         "chance_problem": None}))


def write_run(run, *, hours=(20, 22, 21), collect_hours=10, usd=300.0, **kw):
    for n in (1, 2, 3):
        write_experiment(run, n, **kw)
    t, marks = 0.0, []
    for n, h in zip((1, 2, 3), hours):
        for stage, dur in (("design", 2.0), ("collect", collect_hours), ("loop", h - 2.0)):
            marks.append({"experiment": n, "stage": stage, "event": "started", "at": t * 3600, "job": "1"})
            t += dur
            marks.append({"experiment": n, "stage": stage, "event": "finished", "at": t * 3600, "job": "1"})
    (run / "stage_times.jsonl").write_text("\n".join(json.dumps(m) for m in marks) + "\n")
    from src.runtime.token_usage import UsageRecord

    rec = UsageRecord(source="rsa_open:designer", backend="opencode", model="m", input_tokens=10, output_tokens=10,
                      cost_usd=usd)
    (run / "token_usage.jsonl").write_text(json.dumps(rec.__dict__) + "\n")


def test_a_chain_that_meets_every_criterion_passes(tmp_path):
    from src.rsa_open.gate import evaluate, lines

    write_run(tmp_path)
    report = evaluate(tmp_path)
    assert report["passed"], "\n".join(lines(report))
    c = report["5_cost_and_time"]
    assert c["running_hours"] == pytest.approx(63)  # 3 x (2 h design + 10 h collection + loop) less collection
    item = report["experiments"][1]["3_information"]
    assert item["incumbent"] == "a" and item["ratio"] > 4 and item["carried"]["furthest"] == "b"


@pytest.mark.parametrize("change,item", [
    (dict(ahead=0.5), "3_information"),            # the incumbent is no better than chance
    (dict(exclusion_rate=0.3), "2_data"),
    (dict(same=0.15), "2_data"),                   # more than 10% gave the same answer every trial
    (dict(trusted=("a",)), "4_loop"),              # one trusted theory at the end
])
def test_each_experiment_criterion_fails_on_its_own(tmp_path, change, item):
    from src.rsa_open.gate import evaluate

    write_run(tmp_path, **change)
    report = evaluate(tmp_path)
    assert not report["passed"] and not report["experiments"][1][item]["passed"]
    others = [k for k, v in report["experiments"][1].items() if k != item]
    assert all(report["experiments"][1][k]["passed"] for k in others)


def test_admissions_count_over_the_three_experiments(tmp_path):
    from src.rsa_open.gate import evaluate

    write_run(tmp_path, admitted=())
    assert not evaluate(tmp_path)["4b_admissions"]["passed"]
    write_experiment(tmp_path, 2, admitted=("new",))  # one admission in experiment 2 is enough
    assert evaluate(tmp_path)["passed"]


def test_running_time_and_agent_spend_are_held_to_one_and_a_half_times_the_estimate(tmp_path):
    from src.rsa_open.gate import ESTIMATE_AGENT_USD, ESTIMATE_HOURS, TIME_FACTOR, evaluate

    write_run(tmp_path, hours=(40, 40, 40), collect_hours=24)  # 120 h of design and loops; collection excluded
    c = evaluate(tmp_path)["5_cost_and_time"]
    assert c["running_hours"] == pytest.approx(120) and 120 > TIME_FACTOR * ESTIMATE_HOURS and not c["passed"]
    write_run(tmp_path, usd=TIME_FACTOR * ESTIMATE_AGENT_USD + 1)
    assert not evaluate(tmp_path)["5_cost_and_time"]["passed"]


def test_a_smoke_test_or_a_simulated_collection_is_never_judged(tmp_path):
    from src.rsa_open.gate import evaluate

    write_run(tmp_path)
    path = tmp_path / "experiment1" / "data" / "participants.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "smoke_test": True}))
    assert not evaluate(tmp_path)["experiments"][1]["2_data"]["passed"]
    path.unlink()
    assert not evaluate(tmp_path)["experiments"][1]["2_data"]["passed"]


def test_the_chain_logs_when_each_stage_starts_and_finishes(tmp_path):
    from src.rsa_open.chain import Chain, ChainConfig

    chain = Chain(ChainConfig(chain="chain_0", seeds_dir=tmp_path, literature=None, out_dir=tmp_path / "run",
                              ground_truth=None), designer_spawn=None, theorist_spawn=None)
    done = tmp_path / "run" / "done.json"
    assert chain._stage("loop", 2, done, lambda: done.write_text("{}") and "ran") == "ran"
    assert chain._stage("loop", 2, done, lambda: "resumed") == "resumed"  # output exists: not logged again
    marks = [json.loads(x) for x in (tmp_path / "run" / "stage_times.jsonl").read_text().splitlines()]
    assert [(m["experiment"], m["stage"], m["event"]) for m in marks] == [(2, "loop", "started"), (2, "loop", "finished")]
