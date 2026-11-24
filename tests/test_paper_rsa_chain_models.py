"""``scripts/paper/rsa_chain_models.py`` without its Sherlock-only data: which
models of each chain it scores, and the checks that refuse other data or
missing records. (Its fit and score need the pooled and pre-registered trials,
which are not in the repository.)"""

from __future__ import annotations

import collections

import pytest

from scripts.paper import rsa_chain_models as rc


def test_each_chain_scores_its_other_live_models_and_every_model_it_removed():
    models = rc.all_models(rc.Args(mode="score"))
    by_chain = collections.Counter((m["chain"], m["role"]) for m in models.values())
    assert by_chain == {("c0", "live"): 4, ("c0", "removed"): 77, ("c1", "live"): 11,
                        ("c1", "removed"): 55, ("c2", "live"): 11, ("c2", "removed"): 66}
    assert "c0:boundary_avoidable_ambiguity_listener" not in models  # the final: already scored
    assert "c0:rsa_l2" not in models  # a starting model: scored as part of the bar
    retired = [m for m in models.values() if m.get("outcome", "").startswith("retired")]
    assert len(retired) == 6 + 8 + 8  # retired at carry-overs; the ledger ends them "admitted"
    assert all(m["file"].exists() for m in models.values())


def test_the_settings_are_the_bars():
    rc.check_bar_files(rc.Args(mode="score"))


def test_fitting_refuses_pooled_data_other_than_the_bars(tmp_path):
    other = tmp_path / "pooled.csv"
    other.write_text("not the pooled data\n")
    with pytest.raises(RuntimeError, match="is not the pooled data the bar was fitted on"):
        rc.fit_shard(rc.Args(mode="fit", pooled=other, work_dir=tmp_path / "work"))


def test_scoring_refuses_until_every_model_has_a_record(tmp_path):
    with pytest.raises(RuntimeError, match="224 of 224 chain models have no current record"):
        rc.score(rc.Args(mode="score", test=tmp_path / "unused.csv", work_dir=tmp_path / "work"))


# --- the promotion campaign's existing-data loop (--model-set campaign) -----------------------


def _campaign():
    return rc.Args(mode="score", model_set="campaign")


def test_the_campaign_path_is_each_replicates_incumbent_at_every_step():
    paths = rc.campaign_paths(_campaign())
    assert sorted(paths) == ["real_rep1", "real_rep2", "real_rep3"]
    assert [len(p) for p in paths.values()] == [10, 10, 10]
    assert all(p[0] == {"step": 0, "round": -1, "best_model": "rsa_l2"} for p in paths.values())
    assert paths["real_rep2"][-1]["best_model"] == "surprisal_isolated_graded_costly_l5"


def test_a_campaign_incumbent_with_a_bar_models_source_reuses_its_score():
    # rsa_l2 (every replicate's start) and two of real_rep1's incumbents are, byte for byte, models
    # the pre-registered analysis scored (the bar records each source's sha256).
    incumbents = rc.campaign_incumbents(_campaign())
    reused = {label: info["reuse"] for label, info in incumbents.items() if info["reuse"]}
    assert reused == {
        "real_rep1:rsa_l2": ["promoted:rsa_l2", "seed:rsa_l2"],
        "real_rep2:rsa_l2": ["promoted:rsa_l2", "seed:rsa_l2"],
        "real_rep3:rsa_l2": ["promoted:rsa_l2", "seed:rsa_l2"],
        "real_rep1:rsa_l2_singleton_distinct_feat_color_l0": ["promoted:rsa_l2_singleton_distinct_feat_color_l0"],
        "real_rep1:rsa_l2_singleton_feat_color_valence_l0": ["promoted:rsa_l2_singleton_feat_color_valence_l0"],
    }
    assert len(incumbents) == 7 + 8 + 7  # each replicate's distinct incumbents


def test_the_campaign_fits_only_the_incumbents_the_bar_did_not_score():
    models = rc.all_models(_campaign())
    assert collections.Counter(m["chain"] for m in models.values()) == {
        "real_rep1": 4, "real_rep2": 7, "real_rep3": 6}
    assert "real_rep1:rsa_l2" not in models
    assert all(m["file"].exists() for m in models.values())


def test_campaign_scoring_refuses_until_every_model_has_a_record(tmp_path):
    args = rc.Args(mode="score", model_set="campaign", test=tmp_path / "unused.csv", work_dir=tmp_path / "work")
    with pytest.raises(RuntimeError, match="17 of 17 campaign models have no current record"):
        rc.score(args)


def test_the_campaigns_sums_give_each_replicates_incumbent_its_lpd_at_every_step(tmp_path):
    import json
    import types

    args = rc.Args(mode="score", model_set="campaign", test=tmp_path / "test.csv", out=tmp_path / "out.json")
    args.test.write_text("trials\n")
    prereg = json.loads(args.prereg_analysis.read_text())
    bar = json.loads((args.live_root / "bar_pool_preds.json").read_text())
    fitted = {label: -10000.0 - k for k, label in enumerate(sorted(rc.all_models(args)))}
    del fitted["real_rep3:singleton_isolation_l2"]
    unscored = {"real_rep3:singleton_isolation_l2": "not converged (r_hat)"}
    trials = types.SimpleNamespace(choices=[0] * prereg["n_trials"])
    rc.write_campaign(args, fitted, unscored, prereg, bar, trials, 1e-9)
    out = json.loads(args.out.read_text())
    assert out["models"]["real_rep1:rsa_l2"]["lpd"] == prereg["lpd"]["seed:rsa_l2"]  # reused
    assert "byte for byte" in out["models"]["real_rep1:rsa_l2"]["scored"]
    assert out["models"]["real_rep2:distinctive_object_l2"]["lpd"] == fitted["real_rep2:distinctive_object_l2"]
    assert out["models"]["real_rep3:singleton_isolation_l2"]["lpd"] is None
    rep1 = out["replicates"]["real_rep1"]
    assert [st["step"] for st in rep1] == list(range(10))
    assert rep1[5] == {"step": 5, "round": 4, "best_model": "rsa_l2_singleton_distinct_features_l0",
                       "lpd": fitted["real_rep1:rsa_l2_singleton_distinct_features_l0"]}
    assert rep1[-1]["lpd"] == prereg["lpd"]["promoted:rsa_l2_singleton_feat_color_valence_l0"]
    assert out["best_promoted_lpd"] == prereg["lpd"][prereg["best_promoted"]]
