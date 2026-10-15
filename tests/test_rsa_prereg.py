"""The pre-registered final experiment: frozen predictions, the primary test, power, design, run."""

import json

import numpy as np
import pandas as pd
import pytest

from src.rsa.design.run import KIND_DIMENSIONS, parse_quotas
from src.rsa.loop.novelty import plain_pool, pool_digest, pool_record
from src.rsa.outer.run import DESIGN_QUOTAS
from src.rsa.prereg import analysis, power
from src.rsa.prereg.predictions import WIDTH, load_predictions


@pytest.fixture(scope="module")
def pool():
    return plain_pool()


def valid_slots(pool):
    v = np.zeros((len(pool), WIDTH), dtype=bool)
    for i, ctx in enumerate(pool):
        for j, c in enumerate(ctx.choice_classes()):
            v[i, j] = c == j
    return v


def random_probs(pool, rng, concentration=1.0):
    v = valid_slots(pool)
    p = np.where(v, rng.gamma(concentration, size=v.shape), 0.0)
    p = np.where(v, p + 0.02, 0.0)  # every class possible (a lapse)
    return p / p.sum(-1, keepdims=True)


def write_preds(path, pool, models):
    """A predictions file in pool_fit's format from (displays, WIDTH) class probabilities."""
    n_entries = [len(c.objects) for c in pool]
    doc = dict(note="test", pool=dict(size=len(pool), digest=pool_digest(pool), n_entries=n_entries,
                                       choice_classes=[list(c.choice_classes()) for c in pool],
                                       contexts=[pool_record(c) for c in pool]),
               models={name: dict(probs=[round(float(x), 6) for i, row in enumerate(p) for x in row[: n_entries[i]]])
                       for name, p in models.items()})
    path.write_text(json.dumps(doc))
    return path


@pytest.fixture(scope="module")
def preds(pool, tmp_path_factory):
    """Three finals, three promoted seeds (one close to final a), two starting models."""
    rng = np.random.default_rng(0)
    base = {n: random_probs(pool, rng) for n in ("a", "b", "c", "promoted:p1", "promoted:p2", "seed:s1", "seed:s2")}
    v = valid_slots(pool)
    near = np.where(v, base["a"] * np.exp(rng.normal(0, 0.25, base["a"].shape)), 0.0)
    base["promoted:near_a"] = near / near.sum(-1, keepdims=True)
    d = tmp_path_factory.mktemp("preds")
    finals = write_preds(d / "finals.json", pool, {k: base[k] for k in ("a", "b", "c")})
    bar = write_preds(d / "bar.json", pool, {k: v for k, v in base.items() if ":" in k})
    return load_predictions([finals, bar])


def test_predictions_load_into_the_pools_layout_and_find_displays(preds, pool):
    assert preds.names[:3] == ["a", "b", "c"] and len(preds.names) == 8
    assert preds.probs.shape == (8, len(pool), WIDTH)
    assert np.allclose(preds.probs.sum(-1), 1) and not preds.probs[:, ~preds.valid].any()
    assert all(preds.display(pool[i]) == i for i in (0, 100, len(pool) - 1))
    assert len(preds.sha256) == 2


def test_predictions_from_another_pool_or_with_a_repeated_model_are_refused(tmp_path, pool):
    p = random_probs(pool, np.random.default_rng(1))
    path = write_preds(tmp_path / "x.json", pool, {"a": p})
    doc = json.loads(path.read_text())
    doc["pool"]["digest"] = "0" * 16
    (tmp_path / "bad.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="another pool"):
        load_predictions([tmp_path / "bad.json"])
    with pytest.raises(ValueError, match="two predictions files"):
        load_predictions([path, path])


def test_holm_steps_down_and_stops_at_the_first_failure():
    assert analysis.holm({"a": 0.01, "b": 0.02, "c": 0.04}) == {"a": True, "b": True, "c": True}
    assert analysis.holm({"a": 0.01, "b": 0.03, "c": 0.04}) == {"a": True, "b": False, "c": False}
    assert analysis.holm({"a": 0.02, "b": 0.001, "c": 0.9}) == {"a": True, "b": True, "c": False}


def simulated_trials(preds, truth, design, m, rng):
    contexts, choices, units = [], [], []
    for g, i in enumerate(design):
        ctx = preds.pool[i]
        for c in rng.choice(WIDTH, size=m, p=truth[i]):
            contexts.append(ctx)
            choices.append(int(c))
            units.append(g)
    return contexts, choices, np.asarray(units)


def test_the_primary_test_picks_the_best_promoted_seed_after_the_fact_and_finds_the_true_final(preds):
    rng = np.random.default_rng(2)
    design = rng.choice(len(preds.pool), 150, replace=False)
    contexts, choices, units = simulated_trials(preds, preds.model("a"), design, 30, rng)
    lp = analysis.trial_lpd(preds, contexts, choices)
    out = analysis.primary(dict(zip(preds.names, lp)), units, ["a", "b", "c"])
    assert out["best_promoted"] == "promoted:near_a"  # the strongest bar model on these data
    assert out["primary"]["a"]["holm_reject"] and out["primary"]["a"]["z"] > 3
    assert not out["primary"]["b"]["holm_reject"] and not out["all_finals_beat_the_bar"]
    assert out["best_starting"].startswith("seed:")
    assert set(out["each_bar_model"]["a"]) == {n for n in preds.names if ":" in n}
    # The SE is display-clustered: as src.models.clustered_se on the per-trial lpd.
    from src.models.clustered_se import cluster_dse
    ia, ib = preds.names.index("a"), preds.names.index("promoted:near_a")
    assert out["primary"]["a"]["se"] == pytest.approx(cluster_dse(lp[ia], lp[ib], units))


def test_a_display_outside_the_pool_or_a_choice_a_model_rules_out_raises(preds, pool):
    from src.rsa.context import Context

    off = Context(objects=((1, 0, 0, 0, 0), (0, 1, 0, 0, 0)), feature_names=tuple("abcde"), utterance=0)
    with pytest.raises(ValueError, match="not in the design pool"):
        analysis.trial_lpd(preds, [off], [0])


def test_calibration_gives_the_final_exactly_delta_nats_a_trial(preds):
    design = np.arange(0, 1500, 7)
    pf, pb, v = (preds.model("a")[design], preds.model("promoted:near_a")[design], preds.valid[design])
    lam, capped = power.calibrate(pf, pb, v, 0.004)
    truth = lam * pf + (1 - lam) * pb
    gap = np.where(v, np.log(np.where(v, pf, 1)) - np.log(np.where(v, pb, 1)), 0)
    assert (truth * gap).sum(-1).mean() == pytest.approx(0.004, rel=1e-6) and not capped
    assert power.calibrate(pf, pb, v, 10.0) == (1.0, True)


def test_random_designs_meet_the_quotas_and_leave_out_what_they_are_told(pool):
    rng = np.random.default_rng(3)
    exclude = list(range(0, 1500, 3))
    picks = power.random_design(pool, DESIGN_QUOTAS, 52, rng, exclude=exclude)
    assert len(set(picks)) == 52 and not set(picks) & set(exclude)
    for q in parse_quotas(DESIGN_QUOTAS, pool, 52):
        assert q.members[picks].sum() >= q.minimum
    assert {KIND_DIMENSIONS["objects"](pool[i]) for i in picks} >= {"2", "3"}


def test_power_is_high_when_a_final_is_true_and_near_alpha_when_it_has_no_advantage(preds):
    rng = np.random.default_rng(4)
    design = power.random_design(preds.pool, DESIGN_QUOTAS, 104, rng)
    strong = power.simulate(preds, ["a", "b", "c"], "a", design, 40, 200, "model", rng)
    assert strong["power"] > 0.95 and strong["nearest_promoted"] == "promoted:near_a" and strong["lam"] == 1.0
    null = power.simulate(preds, ["a", "b", "c"], "a", design, 40, 400, "calibrated:0", rng)
    assert null["power"] < 0.1


def test_the_empirical_scaling_grows_with_the_square_root_of_the_displays():
    from scipy.stats import norm
    assert power.empirical_power(norm.isf(0.05), 52, 52, 0.05) == pytest.approx(0.5)
    assert power.empirical_power(1.27, 52, 208, 0.05) > power.empirical_power(1.27, 52, 104, 0.05)
    assert power.responses_per_display(1200, 208, 13, 0.0) == 75


def test_the_power_cli_writes_a_grid_and_the_empirical_scaling(preds, tmp_path):
    files = [p for p in preds.sha256]
    rec = power.main(power.Args(predictions=[__import__("pathlib").Path(f) for f in files], out=tmp_path / "p.json",
                                displays=[52], participants=[200], truths=["model"], n_sims=40, sims_per_design=20))
    assert len(rec["grid"]) == 1 and set(rec["grid"][0]["finals"]) == {"a", "b", "c"}
    assert rec["grid"][0]["responses_per_display"] == power.responses_per_display(200, 52)
    assert set(rec["empirical_scaling"]["power"]["52"]) == set(power.CLAIM2_Z)
    assert json.loads((tmp_path / "p.json").read_text())["grid"][0]["cost_usd"] == 133


@pytest.mark.slow  # runs NUTS
def test_design_then_a_simulated_run_scores_the_frozen_predictions(tmp_path, pool):
    """The design's two halves meet the quotas; the run refuses another design,
    collects simulated people on the pre-registered lists and analyses the
    modelled trials (no catch trials) with the frozen predictions."""
    import shutil
    from pathlib import Path

    import yaml

    from src.rsa.dataset import DEFAULT_TRIALS_CSV
    from src.rsa.prereg import design as prereg_design
    from src.rsa.prereg.run import PreregConfig, PreregRun
    from src.runtime.config import PROJECT_ASSETS_DIR

    seeds = PROJECT_ASSETS_DIR / "rsa_reference" / "seed_models"

    def folder(name, models):
        d = tmp_path / name
        d.mkdir()
        for m in models:
            shutil.copyfile(seeds / f"{m}.py", d / f"{m}.py")
        (d / "models_manifest.yaml").write_text(yaml.safe_dump({"models": [{"name": m} for m in models]}))
        return d

    finals = folder("finals", ["literal_listener", "rsa_l1"])
    starting = folder("starting", ["rsa_l2"])
    promoted = folder("promoted", ["rsa_l1_salience"])
    df = pd.read_csv(DEFAULT_TRIALS_CSV)
    data = tmp_path / "pooled.csv"
    df[df.experiment.isin(["E8_levels", "E9_twins"])].assign(source="pragmods").to_csv(data, index=False)
    fit = dict(num_warmup=150, num_samples=150, num_chains=2)
    rec = prereg_design.main(prereg_design.Args(
        finals=finals, data=data, cache=tmp_path / "cache", out=tmp_path / "design", displays=26, participants=8,
        starting=starting, promoted=promoted, trials=4, n_draws=20, n_scenarios=200, **fit))
    d = json.loads((tmp_path / "design" / "design.json").read_text())
    labels = [t["label"] for t in d["trials"]]
    assert sum(l.startswith("eig_") for l in labels) == 13 and sum(l.startswith("rand_") for l in labels) == 13
    for half in ("eig", "random"):
        for q in parse_quotas(DESIGN_QUOTAS, pool, 13):
            assert rec[half]["quota_counts"][q.name] >= q.minimum
    assert rec["models"][:2] == ["literal_listener", "rsa_l1"] and len(json.loads(
        (tmp_path / "design" / "trial_lists.json").read_text())["lists"]) == 8

    rng = np.random.default_rng(5)
    models = {n: random_probs(pool, rng) for n in ("a", "b", "promoted:p", "seed:s")}
    preds = [write_preds(tmp_path / "preds.json", pool, models)]
    kw = dict(run_dir=tmp_path / "run", design_dir=tmp_path / "design", predictions=preds, participants=8,
              collection="simulated", ground_truth=seeds / "rsa_l1.py", existing_data=data, max_catch_errors=2, **fit)
    with pytest.raises(ValueError, match="not the pre-registered"):
        PreregRun(PreregConfig(design_sha256="0" * 64, **kw))
    with pytest.raises(ValueError, match="not the pre-registered ones"):
        PreregRun(PreregConfig(design_sha256=rec["design_sha256"], predictions_sha256=["0" * 64], **kw))
    PreregRun(PreregConfig(design_sha256=rec["design_sha256"], **kw)).run()
    e1 = tmp_path / "run" / "experiment1"
    out = json.loads((e1 / "prereg_analysis.json").read_text())
    modelled = pd.read_csv(e1 / "data" / "modelled.csv")
    assert (modelled.condition != "catch").all() and out["n_trials"] == len(modelled)
    assert out["finals"] == ["a", "b"] and out["best_promoted"] == "promoted:p" and out["design_sha256"] == rec["design_sha256"]
    assert set(out["by_half"]) == {"eig", "random"}
    assert Path(e1 / "design" / "trial_lists.json").read_bytes() == (tmp_path / "design" / "trial_lists.json").read_bytes()


@pytest.mark.slow  # runs NUTS
def test_pool_fit_writes_predictions_the_analysis_reads(tmp_path):
    from src.rsa.dataset import DEFAULT_TRIALS_CSV
    from src.rsa.prereg import pool_fit
    from src.runtime.config import PROJECT_ASSETS_DIR

    df = pd.read_csv(DEFAULT_TRIALS_CSV)
    data = tmp_path / "pooled.csv"
    df[df.experiment.isin(["E9_twins"])].assign(source="pragmods").to_csv(data, index=False)
    seeds = PROJECT_ASSETS_DIR / "rsa_reference" / "seed_models"
    pool_fit.main(pool_fit.Args(data=data, models=[f"seed:{seeds}"], cache=tmp_path / "cache",
                                out=tmp_path / "bar.json", workers=2, num_warmup=100, num_samples=100, num_chains=2))
    preds = load_predictions([tmp_path / "bar.json"])
    assert sorted(preds.names) == sorted(f"seed:{n}" for n in ("literal_listener", "rsa_l1", "rsa_l2",
                                                               "rsa_l1_salience", "rsa_l1_shared_prior"))
    assert json.loads((tmp_path / "bar.json").read_text())["data"]["sha256"]
