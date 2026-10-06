"""The committed pragmods trial table as model inputs."""

import pandas as pd
import pytest

from src.rsa.dataset import DEFAULT_TRIALS_CSV, context_from_row, load_forced_choice


def test_every_included_forced_choice_trial_with_a_display_becomes_a_context():
    trials = load_forced_choice()
    df = pd.read_csv(DEFAULT_TRIALS_CSV)
    expected = df[(df.dv == "forced_choice") & df.included & (df.objects != "[]")]
    assert len(trials.contexts) == len(trials.choices) == len(expected)
    assert all(0 <= c < ctx.shape[0] for ctx, c in zip(trials.contexts, trials.choices))


def test_the_simple_game_with_glasses_is_the_canonical_display():
    trials = load_forced_choice(experiments=["E8_levels"])
    simple_glasses = [
        c for c in trials.contexts if c.objects == ((0, 0), (0, 1), (1, 1)) and c.utterance == 1
    ]
    assert len(simple_glasses) == 46
    assert simple_glasses[0].utterance_names[-1] == "<sink>"


def test_familiarization_becomes_base_rates_and_valence_follows_framing():
    base = load_forced_choice(experiments=["E5_baserate"])
    assert all(abs(sum(c.familiarization) - 1) < 1e-9 for c in base.contexts)
    valence = load_forced_choice(experiments=["E6_valence"])
    assert {c.valence for c in valence.contexts} == {-1, 1}


def test_an_unknown_framing_raises():
    row = pd.read_csv(DEFAULT_TRIALS_CSV).iloc[0].copy()
    row["framing"] = "shouted"
    with pytest.raises(ValueError, match="no valence"):
        context_from_row(row)


def test_an_unknown_experiment_raises():
    with pytest.raises(ValueError, match="no experiments named"):
        load_forced_choice(experiments=["E99"])


def test_a_messages_column_restricts_the_words():
    row = pd.read_csv(DEFAULT_TRIALS_CSV).iloc[0].copy()
    row["objects"] = "[[1,0,0,0,0,1],[1,0,0,1,0,0],[0,1,0,0,1,0]]"
    row["feature_names"] = '["circle","triangle","square","green","red","blue"]'
    row["query"], row["utterance"], row["framing"] = "utterance", 0, "one_word"
    row["familiarization"], row["grayscale"] = float("nan"), float("nan")
    row["messages"] = "[0,1,3,4]"
    assert context_from_row(row).utterance_names == ("circle", "triangle", "green", "red")
