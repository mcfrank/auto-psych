"""The end-of-run prune's clusters: a display within an experimental condition."""

import pandas as pd
import pytest

from src.rsa.loop.orchestrator import cluster_ids


def frame(**overrides):
    base = dict(experiment=["E1", "E2", "E1", "E1"], condition=["a", "a", "a", "b"],
                objects=["[[0,1],[1,1]]"] * 4, query=["utterance"] * 4, utterance=[1, 1, 1, 1],
                familiarization=[None] * 4, grayscale=[None] * 4, framing=["one_word"] * 4)
    base.update(overrides)
    return pd.DataFrame(base)


def test_one_display_in_different_experiments_or_conditions_is_different_clusters():
    ids = cluster_ids(frame())
    assert ids[0] == ids[2]           # same display, experiment and condition
    assert len({ids[0], ids[1], ids[3]}) == 3


def test_different_displays_in_one_condition_are_different_clusters():
    ids = cluster_ids(frame(utterance=[1, 1, 0, 1]))
    assert ids[0] != ids[2]


def test_sources_and_message_sets_separate_clusters_when_present():
    f = frame(source=["a", "a", "b", "a"], messages=["[0,1]", "[0,1]", "[0,1]", "[0]"])
    ids = cluster_ids(f)
    assert ids[0] != ids[2]


def test_responses_without_conditions_raise():
    with pytest.raises(ValueError, match="condition"):
        cluster_ids(frame().drop(columns=["condition"]))


def test_pruned_models_on_the_refinement_menu_rank_by_their_margin():
    from src.rsa.loop.orchestrator import prune_margin

    details = {
        "far": "elpd_diff 120.5 > 2.0 x clustered dse 30.1 vs best",
        "near": "elpd_diff 9.2 > 2.0 x clustered dse 4.0 vs best",
        "capped": "retired by the cap of 8 live models",
    }
    assert sorted(details, key=lambda n: prune_margin(details[n])) == ["near", "far", "capped"]
