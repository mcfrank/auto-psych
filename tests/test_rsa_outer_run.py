"""The RSA outer loop end to end: simulated people, a scripted agent (no API calls)."""

import json
import shutil

import pandas as pd
import pytest
import yaml

from src.rsa.dataset import DEFAULT_TRIALS_CSV
from src.rsa.outer.run import OuterConfig, OuterRun, exclude_on_catch
from src.runtime.config import PROJECT_ASSETS_DIR

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
    df = pd.read_csv(DEFAULT_TRIALS_CSV)
    existing = tmp_path / "existing.csv"
    df[df.experiment.isin(["E8_levels", "E5_baserate"])].assign(source="pragmods").to_csv(existing, index=False)

    def spawner(n, models_dir, responses):
        def agent(cdir, prompt):  # proposes the hidden ground truth's mechanism
            (cdir / "candidate.py").write_text((SEEDS / "rsa_l2.py").read_text())
            (cdir / "hypothesis.md").write_text("Listeners reason at depth 2.")
            (cdir / "model_name.txt").write_text(f"deeper_listener_e{n}\n")
            return True
        return agent

    cfg = OuterConfig(run_dir=tmp_path / "run1", seeds=seeds, existing_data=existing, collection="simulated",
                      ground_truth=SEEDS / "rsa_l2.py", n_experiments=2, participants=12, trials=2, displays=4,
                      n_catch=1, max_catch_errors=1, max_iterations=1, candidate_count=1, cv_folds=2,
                      num_warmup=150, num_samples=150, num_chains=2, n_draws=20, n_scenarios=200,
                      fit_time_limit_sec=None, selection_scope="live")
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
    assert set(pro["models"]) == {"literal_listener", "rsa_l1"} | seeds5
    assert pro["best_seed"] in seeds5 and pro["best_promoted"] is None
    assert pro["models"][pro["best_seed"]]["vs_best_seed"] == {"diff": 0.0, "se": 0.0}
    assert set(pro["live_vs"]) == {"best_seed"}
    # From experiment 2 on, also against the promoted seeds the live phase began with.
    pro2 = json.loads((tmp_path / "run1" / ".private" / "experiment2" / "prospective.json").read_text())
    assert {"promoted:literal_listener", "promoted:rsa_l1"} <= set(pro2["models"])
    assert set(pro2["live_vs"]) == {"best_seed", "best_promoted"}
    # The inner loop ran on everything so far; its live set is experiment 2's input.
    export = json.loads((e1 / "model_loop" / "export.json").read_text())
    assert sorted(p.stem for p in (e2 / "models_input").glob("*.py")) == sorted(export["live"])
    prior2 = pd.read_csv(e2 / "data" / "prior.csv")
    assert len(prior2) == len(pd.read_csv(existing)) + len(r1)
    # Simulated runs record each model's distance to the hidden ground truth.
    private = tmp_path / "run1" / ".private"
    rec = json.loads((private / "experiment1" / "recovery.json").read_text())
    assert rec["ground_truth"] == "rsa_l2" and rec["selection_scope"] == "live" and rec["exported"] == export["best_model"]
    assert set(rec["after"]) == set(export["live"]) and rec["closest_before"] in {"literal_listener", "rsa_l1"}
    # Nothing the agents can read names the ground truth: its fit and the
    # records that name it are private.
    visible = [p for p in (tmp_path / "run1").rglob("*") if p.is_file() and ".private" not in p.parts]
    assert not [p for p in visible if "rsa_l2" in p.name]
    assert not [p for p in visible if p.suffix in (".json", ".csv") and "rsa_l2" in p.read_text()]
    # A finished run resumes to nothing; another configuration is refused.
    OuterRun(cfg, spawner).run()
    cfg.participants = 13
    with pytest.raises(ValueError, match="another configuration"):
        OuterRun(cfg, spawner).run()
