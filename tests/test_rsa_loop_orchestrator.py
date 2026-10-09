"""The RSA loop end to end with a scripted fake agent (no API calls)."""

import json
import shutil

import pandas as pd
import pytest
import yaml

from src.rsa.dataset import DEFAULT_TRIALS_CSV
from src.rsa.fit import FitSettings
from src.rsa.loop.orchestrator import LoopConfig, RSALoop
from src.runtime.config import PROJECT_ASSETS_DIR

pytestmark = pytest.mark.slow  # runs NUTS

SEEDS = PROJECT_ASSETS_DIR / "rsa_reference" / "seed_models"


def write(cdir, source, hypothesis, name=None):
    (cdir / "candidate.py").write_text(source)
    (cdir / "hypothesis.md").write_text(hypothesis)
    if name:
        (cdir / "model_name.txt").write_text(name + "\n")


def fake_agent(cdir, prompt):
    """Slot 1 proposes depth-2 RSA; slot 2 breaks memo then repairs it;
    slot 3 never writes a file."""
    slot = cdir.name
    if slot.startswith("candidate_1"):
        write(cdir, (SEEDS / "rsa_l2.py").read_text(), "Listeners reason at depth 2.", "deeper_listener")
    elif slot == "candidate_2":
        broken = (SEEDS / "rsa_l1_salience.py").read_text().replace("{EPS}", "EPS")
        write(cdir, broken, "A salience prior at the top level.", "salience_listener")
    elif slot == "candidate_2_repair_1":
        assert "REPAIR" in prompt and "MemoError" in prompt
        write(cdir, (SEEDS / "rsa_l1_salience.py").read_text(), "A salience prior at the top level.",
              "salience_listener")
    return True


@pytest.fixture
def loop(tmp_path):
    seeds = tmp_path / "seeds"
    seeds.mkdir()
    for name in ("literal_listener", "rsa_l1"):
        shutil.copyfile(SEEDS / f"{name}.py", seeds / f"{name}.py")
    (seeds / "models_manifest.yaml").write_text(yaml.safe_dump({"models": [
        {"name": "literal_listener", "rationale": "Literal listener."},
        {"name": "rsa_l1", "rationale": "Depth-1 RSA."},
    ]}))
    df = pd.read_csv(DEFAULT_TRIALS_CSV)
    responses = tmp_path / "trials.csv"
    df[df.experiment.isin(["E8_levels", "E5_baserate"])].to_csv(responses, index=False)
    cfg = LoopConfig(
        responses_path=responses, seed_models_dir=seeds, results_dir=tmp_path / "model_loop",
        max_iterations=1, candidate_count=3,
        settings=FitSettings(num_warmup=300, num_samples=300, num_chains=2),
        fit_time_limit_sec=None, cv_folds=2,
    )
    return RSALoop(cfg, fake_agent)


def test_a_round_admits_repairs_retries_and_exports(loop):
    final = loop.run()
    out = loop.dir
    ledger = [json.loads(line) for line in (out / "attempted_hypotheses.jsonl").read_text().splitlines()]
    by = {(e["name"], e["outcome"]) for e in ledger}
    assert ("deeper_listener", "admitted") in by
    assert ("salience_listener", "rejected") in by and ("salience_listener_2", "admitted") in by
    assert any(e["outcome"] == "rejected" and "no candidate.py" in e["detail"] for e in ledger)
    assert (out / "round_1" / "candidate_3_retry_1").exists()
    # Seeds, the deeper model and the repaired salience model were live before pruning.
    history = json.loads((out / "history.json").read_text())
    assert [h["step"] for h in history] == [0, 1, 2]
    assert set(history[1]["standing"]) == {"literal_listener", "rsa_l1", "deeper_listener", "salience_listener_2"}
    # The literal listener loses by far more than 2 clustered SEs and is pruned at the end.
    assert "literal_listener" not in final["standing"]
    # Selection and the prune ran on grouped CV.
    assert all("elpd_cv" in s for s in final["standing"].values())
    assert any("(grouped CV)" in e["detail"] for e in ledger if e["outcome"] == "pruned")
    assert (out / "models" / "pruned" / "literal_listener.py").exists()
    assert (out / "best_model.py").exists()
    export = json.loads((out / "export.json").read_text())
    assert export["best_model"] == final["best_model"]
    report = (out / "report.html").read_text()
    assert "__BUNDLE_JSON__" not in report and "salience_listener_2" in report
    bundle = json.loads((out / "report.bundle.json").read_text())
    assert len(bundle["timeline"]["history"]) == 3
    # Only the incumbent slot's ledger context names a target model.
    contexts = {e["context"] for e in ledger if e["context"].startswith("round 1 candidate 3")}
    assert contexts and all(c.startswith("round 1 candidate 3 refine chosen") and c.split()[-1] in ("chosen", "1")
                            for c in contexts), contexts


class Crash(RuntimeError):
    pass


def test_an_interrupted_run_resumes_from_its_last_scored_step(loop):
    """Round 2 dies in its repair phase, after admitting a model: resuming
    drops what round 2 did and runs it again, then finishes."""
    loop.cfg.max_iterations = 2

    def crashing(cdir, prompt):
        if cdir.parent.name == "round_2" and "repair" in cdir.name:
            raise Crash("the job was killed")
        if cdir.parent.name == "round_2" and cdir.name == "candidate_1":
            write(cdir, (SEEDS / "rsa_l1_shared_prior.py").read_text(), "A shared prior.", "shared_prior_listener")
            return True
        return fake_agent(cdir, prompt)

    loop.spawn = crashing
    with pytest.raises(Crash):
        loop.run()
    out = loop.dir
    ledger_path = out / "attempted_hypotheses.jsonl"
    history = json.loads((out / "history.json").read_text())
    assert [h["round"] for h in history] == [-1, 0]
    at_step = history[-1]["ledger_entries"]
    assert len(ledger_path.read_text().splitlines()) > at_step  # round 2 wrote to the ledger
    # A round-2 admission is in the set before the restart.
    assert (out / "models" / "shared_prior_listener.py").exists()

    resumed = RSALoop(loop.cfg, fake_agent)
    final = resumed.run(resume=True)
    assert (out / "round_2_abandoned_1").exists() and (out / "round_2").exists()
    history = json.loads((out / "history.json").read_text())
    assert [h["round"] for h in history] == [-1, 0, 1, 2]
    ledger = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    # Round 2 ran once in the ledger: its first attempt's lines were dropped.
    assert sum(1 for e in ledger if e["context"] == "round 2 candidate 1 explore") == 1
    assert (out / "export.json").exists() and final["best_model"]
    # The abandoned round's admission was undone (the rerun proposed something else).
    assert not (out / "models" / "shared_prior_listener.py").exists()
    assert "shared_prior_listener" not in {e["name"] for e in ledger}
    with pytest.raises(RuntimeError, match="nothing to resume"):
        RSALoop(loop.cfg, fake_agent).resume()


def test_selection_on_one_source_ranks_and_prunes_on_its_trials_only(tmp_path):
    """Fit on everything, select on one source (the live loop's option): the
    standing's sel_* fields are the out-of-fold lpd summed over that source."""
    seeds = tmp_path / "seeds"
    seeds.mkdir()
    for name in ("literal_listener", "rsa_l1"):
        shutil.copyfile(SEEDS / f"{name}.py", seeds / f"{name}.py")
    (seeds / "models_manifest.yaml").write_text(yaml.safe_dump({"models": [{"name": "literal_listener"},
                                                                           {"name": "rsa_l1"}]}))
    df = pd.read_csv(DEFAULT_TRIALS_CSV)
    df = df[df.experiment.isin(["E8_levels", "E5_baserate"])]
    df = df.assign(source=["live" if e == "E5_baserate" else "pragmods" for e in df.experiment])
    responses = tmp_path / "trials.csv"
    df.to_csv(responses, index=False)
    cfg = LoopConfig(responses_path=responses, seed_models_dir=seeds, results_dir=tmp_path / "loop",
                     max_iterations=0, settings=FitSettings(num_warmup=200, num_samples=200, num_chains=2),
                     fit_time_limit_sec=None, cv_folds=2, selection_source="live")
    final = RSALoop(cfg, fake_agent).run()
    standing = final["standing"]
    for s in standing.values():
        assert s["sel_elpd"] == pytest.approx(s["cv_by_source"]["live"])
    best = max(standing, key=lambda n: standing[n]["sel_elpd"])
    assert final["best_model"] == best
    with pytest.raises(ValueError, match="no trials"):
        RSALoop(LoopConfig(**{**cfg.__dict__, "results_dir": tmp_path / "loop2", "selection_source": "nope"}),
                fake_agent).run()
