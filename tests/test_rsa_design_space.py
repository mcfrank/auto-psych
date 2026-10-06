"""The enumerated space of reference games."""

import pytest

from src.rsa.design_space import _canonical, context_pool, games


@pytest.mark.parametrize(
    "size, n_games",
    [((3, 2), 6), ((3, 3), 10), ((3, 4), 10), ((4, 3), 39), ((4, 4), 97)],
)
def test_game_counts_up_to_relabeling(size, n_games):
    assert len(games(*size)) == n_games


def test_the_pragmods_games_are_in_the_space():
    simple = ((0, 0), (0, 1), (1, 1))  # plain, glasses, hat+glasses
    complex_ = ((0, 0, 1), (0, 1, 1), (1, 1, 0))  # M, GM, HG over hat, glasses, mustache
    twins = ((0, 1, 1), (1, 0, 1), (1, 0, 1))  # GM, HM, HM
    assert _canonical(simple) in games(3, 2)
    assert _canonical(complex_) in games(3, 3)
    assert _canonical(twins) in games(3, 3)


def test_relabeled_games_are_one_class():
    a = ((0, 0), (0, 1), (1, 1))
    b = ((1, 1), (1, 0), (0, 0))  # objects and features both reordered
    assert _canonical(a) == _canonical(b)


def test_games_have_no_word_true_of_nothing_and_no_synonyms():
    for matrix in games(4, 3):
        columns = list(zip(*matrix))
        assert all(any(c) for c in columns)
        assert len(set(columns)) == len(columns)


def test_context_pool_crosses_games_with_words_and_prior_queries():
    pool = context_pool([(3, 2)], include_prior_queries=True)
    assert len(pool) == 6 * (2 + 1)
    assert sum(c.utterance is None for c in pool) == 6
