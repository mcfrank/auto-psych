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
                      fit_time_limit_sec=None)
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
    # Prospective scoring: the models going in and rsa_l2, before any refit.
    pro = json.loads((e1 / "prospective.json").read_text())
    assert set(pro["models"]) == {"literal_listener", "rsa_l1", "rsa_l2"} and pro["reference"] == "rsa_l2"
    assert pro["models"]["rsa_l2"]["diff_vs_reference"] == 0.0
    # The inner loop ran on everything so far; its live set is experiment 2's input.
    export = json.loads((e1 / "model_loop" / "export.json").read_text())
    assert sorted(p.stem for p in (e2 / "models_input").glob("*.py")) == sorted(export["live"])
    prior2 = pd.read_csv(e2 / "data" / "prior.csv")
    assert len(prior2) == len(pd.read_csv(existing)) + len(r1)
    # A finished run resumes to nothing; another configuration is refused.
    OuterRun(cfg, spawner).run()
    cfg.participants = 13
    with pytest.raises(ValueError, match="another configuration"):
        OuterRun(cfg, spawner).run()
