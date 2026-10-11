"""The RSA outer loop end to end: simulated people, a scripted agent (no API calls)."""

import json
import shutil

import numpy as np
import pandas as pd
import pytest
import yaml

from src.rsa.dataset import DEFAULT_TRIALS_CSV
from src.rsa.outer.run import OuterConfig, OuterRun, exclude_on_catch
from src.runtime.config import PROJECT_ASSETS_DIR
from src.runtime.token_usage import record_usage

SEEDS = PROJECT_ASSETS_DIR / "rsa_reference" / "seed_models"


def test_participants_who_miss_catch_trials_are_excluded():
    cov = lambda catch, ok: json.dumps({"is_catch": catch, "catch_correct": ok})  # noqa: E731
    rows = pd.DataFrame({"participant_id": ["a", "a", "b", "b", "c"],
                         "covariates": [cov(True, True), cov(False, None), cov(True, False), cov(False, None),
                                        cov(True, False)]})
    kept, record = exclude_on_catch(rows, 0)
    assert list(kept.participant_id) == ["a", "a"] and record["excluded"] == ["b", "c"]
    assert exclude_on_catch(rows, 1)[1]["excluded"] == []


@pytest.mark.slow  # runs NUTS
def test_two_experiments_design_collect_score_and_carry_the_live_set(tmp_path):
    seeds = tmp_path / "seeds"
    seeds.mkdir()
    for n in ("literal_listener", "rsa_l1"):
        shutil.copyfile(SEEDS / f"{n}.py", seeds / f"{n}.py")
    (seeds / "models_manifest.yaml").write_text(yaml.safe_dump({"models": [{"name": "literal_listener"},
                                                                           {"name": "rsa_l1"}]}))
    # This chain began with a share of the promoted seeds; its bar is all of them.
    promoted = tmp_path / "promoted"
    shutil.copytree(seeds, promoted)
    shutil.copyfile(SEEDS / "rsa_l1_salience.py", promoted / "rsa_l1_salience.py")
    (promoted / "models_manifest.yaml").write_text(yaml.safe_dump({"models": [
        {"name": "literal_listener"}, {"name": "rsa_l1"}, {"name": "rsa_l1_salience"}]}))
    df = pd.read_csv(DEFAULT_TRIALS_CSV)
    existing = tmp_path / "existing.csv"
    df[df.experiment.isin(["E8_levels", "E9_twins", "E5_baserate"])].assign(source="pragmods").to_csv(existing, index=False)

    def spawner(n, models_dir, responses):
        def agent(cdir, prompt):  # proposes the hidden ground truth's mechanism
            if cdir.name.startswith("critique"):  # the critique step (main's CriticAL)
                (cdir / "test_stats").mkdir(parents=True, exist_ok=True)
                (cdir / "test_stats" / "share_first.py").write_text(
                    "# name: share_first\n# description: Share choosing object 0.\n"
                    "def test_statistic(df):\n    return float((df['choice'] == 0).mean())\n")
                return True
            record_usage(source="rsa:candidate", backend="opencode", model="m", input_tokens=100, cost_usd=0.25)
            (cdir / "candidate.py").write_text((SEEDS / "rsa_l2.py").read_text())
            (cdir / "hypothesis.md").write_text("Listeners reason at depth 2.")
            (cdir / "model_name.txt").write_text(f"deeper_listener_e{n}\n")
            return True
        return agent

    cfg = OuterConfig(run_dir=tmp_path / "run1", seeds=seeds, promoted=promoted, existing_data=existing, collection="simulated",
                      ground_truth=SEEDS / "rsa_l2.py", n_experiments=2, participants=12, trials=2, displays=4,
                      n_catch=1, max_catch_errors=1, max_iterations=1, candidate_count=1, cv_folds=2,
                      num_warmup=150, num_samples=150, num_chains=2, n_draws=20, n_scenarios=200,
                      fit_time_limit_sec=None, selection_scope="guarded")
    OuterRun(cfg, spawner).run()
    e1, e2 = tmp_path / "run1" / "experiment1", tmp_path / "run1" / "experiment2"
    # Experiment 1 starts from the seeds; its design has 4 displays, one list per person.
    assert sorted(p.stem for p in (e1 / "models_input").glob("*.py")) == ["literal_listener", "rsa_l1"]
    lists = json.loads((e1 / "design" / "trial_lists.json").read_text())
    assert len(lists["lists"]) == 12 and len(json.loads((e1 / "design" / "design.json").read_text())["trials"]) == 4
    # Collected rows are this experiment's, ids unique across experiments.
    r1, r2 = (pd.read_csv(e / "data" / "responses.csv") for e in (e1, e2))
    assert set(r1.experiment) == {"run1_e1"} and set(r2.experiment) == {"run1_e2"}
    assert not set(r1.participant_id) & set(r2.participant_id)
    # Prospective scoring: the models going in against the best of the five
    # starting models, fitted to the same earlier data, before any refit.
    pro = json.loads((tmp_path / "run1" / ".private" / "experiment1" / "prospective.json").read_text())
    seeds5 = {f"seed:{n}" for n in ("literal_listener", "rsa_l1", "rsa_l2", "rsa_l1_salience", "rsa_l1_shared_prior")}
    promoted3 = {f"promoted:{n}" for n in ("literal_listener", "rsa_l1", "rsa_l1_salience")}
    assert set(pro["models"]) == {"literal_listener", "rsa_l1"} | seeds5 | promoted3
    assert pro["best_seed"] in seeds5 and pro["best_promoted"] in promoted3
    assert pro["models"][pro["best_seed"]]["vs_best_seed"] == {"diff": 0.0, "se": 0.0}
    assert set(pro["live_vs"]) == {"best_seed", "best_promoted"}
    # And from experiment 2 on, against every promoted seed.
    pro2 = json.loads((tmp_path / "run1" / ".private" / "experiment2" / "prospective.json").read_text())
    assert promoted3 <= set(pro2["models"])
    assert set(pro2["live_vs"]) == {"best_seed", "best_promoted"}
    # The inner loop ran on everything so far; its live set is experiment 2's input.
    export = json.loads((e1 / "model_loop" / "export.json").read_text())
    # One model per hypothesis the live displays can tell apart goes on.
    carry = json.loads((e2 / "carry.json").read_text())
    assert sorted(p.stem for p in (e2 / "models_input").glob("*.py")) == sorted(carry["kept"])
    assert set(carry["kept"]) <= set(export["live"]) and set(carry["order"]) == set(export["live"])
    prior2 = pd.read_csv(e2 / "data" / "prior.csv")
    # The live phase fits plain displays only: E5's familiarization trials are left out, on record.
    plain = pd.read_csv(tmp_path / "run1" / ".private" / "existing_plain.csv")
    assert len(prior2) == len(plain) + len(r1) and "E5_baserate" not in set(plain.experiment)
    scope = json.loads((tmp_path / "run1" / "existing_scope.json").read_text())
    assert scope["scope"] == "plain" and set(scope["left_out"]) == {"pragmods|E5_baserate"}
    # The agents are told, and experiment 2's ledger continues experiment 1's.
    assert "Scope: plain displays" in next((e1 / "model_loop").glob("round_1/candidate_1/CONTEXT.md")).read_text()
    # Every round critiques its incumbent first, and the candidates get it.
    for e in (e1, e2):
        rounds = [h for h in json.loads((e / "model_loop" / "history.json").read_text())
                  if 0 <= h["round"] < cfg.max_iterations]  # not the seed step (-1) or the end-of-run prune
        assert rounds and all(h["critique"]["status"] == "critiqued" for h in rounds)
    assert "## critiques.md" not in (e1 / "model_loop" / "round_1" / "candidate_1" / "CONTEXT.md").read_text()
    assert (e1 / "model_loop" / "round_1" / "candidate_1" / "critiques.md").exists()
    l1 = (e1 / "model_loop" / "attempted_hypotheses.jsonl").read_text().splitlines()
    l2 = (e2 / "model_loop" / "attempted_hypotheses.jsonl").read_text().splitlines()
    assert l1 and l2[:len(l1)] == l1
    # Claim 2's test: the model experiment 1's loop exported, committed before experiment 2's data.
    assert pro["committed"] is None
    assert pro2["committed"] == export["best_model"] and set(pro2["committed_vs"]) == {
        "best_seed", "best_promoted", "each_bar_model"}
    assert set(pro2["committed_vs"]["each_bar_model"]) == {k for k in pro2["models"] if ":" in k}
    # Simulated runs record each model's distance to the hidden ground truth.
    private = tmp_path / "run1" / ".private"
    rec = json.loads((private / "experiment1" / "recovery.json").read_text())
    assert rec["ground_truth"] == "rsa_l2" and rec["selection_scope"] == "guarded" and rec["exported"] == export["best_model"]
    assert set(rec["after"]) == set(export["live"]) and rec["closest_before"] in {"literal_listener", "rsa_l1"}
    assert {"kl_pool", "kl_design", "rmse_pool"} == set(rec["exported_distance"])
    # The design also aims at the bar models (the starting models and promoted
    # seeds), with the ground truth withheld from it, unnamed.
    eig = json.loads((e1 / "design" / "eig.json").read_text())
    assert "bar:rsa_l1_salience" in eig["models"] and "bar:rsa_l2" not in eig["models"] and eig["n_withheld"] == 1
    assert sum(p for p, g in zip(eig["prior"], eig["model_groups"]) if g == "carried") == pytest.approx(0.5)
    # Nothing the agents can read names the ground truth: its fit and the
    # records that name it are private.
    visible = [p for p in (tmp_path / "run1").rglob("*") if p.is_file() and ".private" not in p.parts]
    assert not [p for p in visible if "rsa_l2" in p.name]
    assert not [p for p in visible if p.suffix in (".json", ".csv") and "rsa_l2" in p.read_text()]
    # The agents' spend is recorded per experiment and for the run.
    usage = json.loads((tmp_path / "run1" / "token_usage_summary.json").read_text())
    assert set(usage["experiments"]) == {"experiment1", "experiment2"}
    assert usage["total"]["cost_usd"] == pytest.approx(0.25 * usage["total"]["n_calls"]) and usage["total"]["n_calls"] >= 2
    assert (e1 / "model_loop" / "token_usage_summary.json").exists()
    # A finished run resumes to nothing; another configuration is refused.
    OuterRun(cfg, spawner).run()
    cfg.participants = 13
    with pytest.raises(ValueError, match="another configuration"):
        OuterRun(cfg, spawner).run()


def test_the_recovery_kl_is_a_mean_per_display_of_flattened_class_probabilities():
    from src.rsa.outer.run import mean_kl

    one = [0.5, 0.5]  # a two-class display
    two = [0.25, 0.25, 0.5]  # a three-class display
    truth = np.array(one + two)
    model = np.array([0.25, 0.75] + two)  # wrong on the first display only
    first = 0.5 * np.log(0.5 / 0.25) + 0.5 * np.log(0.5 / 0.75)
    assert mean_kl(truth, model, 2) == pytest.approx(first / 2)
    assert mean_kl(truth, truth, 2) == 0.0


def test_a_run_launched_for_one_experiment_continues_in_place_with_more(tmp_path, monkeypatch):
    """PI 2026-10-11: the campaign runs experiment 1 first, then is resumed with
    n_experiments 3. How far a run goes is operational; what it studies is not."""
    done = []
    for stage in ("design", "collect", "prospective", "model_loop", "recovery"):
        monkeypatch.setattr(OuterRun, stage, lambda self, n, _s=stage: done.append((_s, n)))
    base = dict(run_dir=tmp_path / "run", seeds=tmp_path, existing_data=tmp_path / "e.csv", collection="live")
    OuterRun(OuterConfig(**base, n_experiments=1), lambda *a: None).run()
    assert {n for _, n in done} == {1}
    done.clear()
    OuterRun(OuterConfig(**base, n_experiments=3), lambda *a: None).run()
    assert [n for s, n in done if s == "design"] == [1, 2, 3]  # experiment 1's stages are kept, by their files
    with pytest.raises(ValueError, match="another configuration"):
        OuterRun(OuterConfig(**base, n_experiments=3, participants=100), lambda *a: None).run()
