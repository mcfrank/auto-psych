"""Reference-game contexts: validation, utterance sets, the sink, shape groups."""

import numpy as np
import pytest

from src.rsa.context import Context, group_by_shape


def simple_context(utterance=1):
    # The pragmods "simple" game: a plain face, one with glasses, one with a
    # hat and glasses; words are hat (0) and glasses (1).
    return Context(
        objects=((0, 0), (0, 1), (1, 1)),
        feature_names=("hat", "glasses"),
        utterance=utterance,
    )


def test_utterances_are_present_features_plus_a_sink_for_featureless_objects():
    ctx = simple_context()
    assert ctx.utterance_names == ("hat", "glasses", "<sink>")
    lex = ctx.lexicon()
    np.testing.assert_array_equal(
        lex, [[0, 0, 1], [0, 1, 1], [1, 0, 0]]
    )
    assert ctx.shape == (3, 3)


def test_no_sink_when_every_object_has_a_true_word():
    ctx = Context(objects=((1, 0), (1, 1)), feature_names=("a", "b"), utterance=0)
    assert ctx.utterance_names == ("a", "b")
    assert ctx.shape == (2, 2)


def test_a_feature_no_object_has_is_not_an_utterance_and_cannot_be_said():
    ctx = Context(
        objects=((1, 0, 0), (1, 1, 0)), feature_names=("a", "b", "c"), utterance=1
    )
    assert ctx.utterance_names == ("a", "b")
    with pytest.raises(ValueError, match="true of no object"):
        Context(objects=((1, 0, 0), (1, 1, 0)), feature_names=("a", "b", "c"), utterance=2)


def test_prior_query_has_no_utterance():
    ctx = simple_context(utterance=None)
    arrays = ctx.arrays()
    assert float(arrays.is_prior) == 1.0
    assert int(arrays.utterance) == 0


def test_utterance_index_maps_feature_to_utterance_position():
    ctx = Context(
        objects=((0, 0, 1), (0, 1, 1)), feature_names=("a", "b", "c"), utterance=2
    )
    # Feature "a" is absent, so "c" is utterance 1.
    assert int(ctx.arrays().utterance) == 1


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(objects=(), feature_names=("a",), utterance=None), "at least one object"),
        (dict(objects=((1,), (1, 0)), feature_names=("a",), utterance=0), "rectangular"),
        (dict(objects=((2,),), feature_names=("a",), utterance=0), "0/1"),
        (dict(objects=((1,),), feature_names=("a",), utterance=0, valence=3), "valence"),
        (
            dict(objects=((1,), (0,)), feature_names=("a",), utterance=0, familiarization=(0.5,)),
            "one per object",
        ),
    ],
)
def test_invalid_contexts_raise(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Context(**kwargs)


def test_group_by_shape_stacks_trials_and_keeps_their_indices():
    a = simple_context(1)
    b = Context(objects=((1, 0), (1, 1)), feature_names=("a", "b"), utterance=1)
    groups = group_by_shape([a, b, simple_context(0)])
    assert set(groups) == {(3, 3), (2, 2)}
    g = groups[(3, 3)]
    assert list(g.indices) == [0, 2]
    assert g.arrays.lex.shape == (2, 3, 3)
    np.testing.assert_array_equal(np.asarray(g.arrays.utterance), [1, 0])
    assert list(groups[(2, 2)].indices) == [1]
