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
