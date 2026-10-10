"""The live phase's scope: plain displays only (PI 2026-10-10)."""

from src.rsa.context import model_key
from src.rsa.dataset import DEFAULT_TRIALS_CSV, is_plain, load_forced_choice, write_plain_trials
from src.rsa.loop.brief import DEFAULT_RSA_LENSES, PLAIN_RSA_LENSES
from src.rsa.loop.novelty import novelty_pool, plain_pool


def test_the_plain_trials_leave_out_exactly_the_framing_exposure_and_colour_experiments(tmp_path):
    record = write_plain_trials(DEFAULT_TRIALS_CSV, tmp_path / "plain.csv")
    assert {k.split("|")[1] for k in record["left_out"]} == {"E5_baserate", "E6_valence", "E7_color",
                                                            "color_prior_rerun"}
    kept = load_forced_choice(tmp_path / "plain.csv")
    assert len(kept.contexts) == record["n_kept"] and all(is_plain(c) for c in kept.contexts)
    assert record["n_kept"] + record["n_left_out"] == len(load_forced_choice(DEFAULT_TRIALS_CSV).contexts)


def test_the_plain_pool_and_lenses():
    pool = plain_pool()
    assert all(is_plain(c) for c in pool)
    # every plain display of the run-2 novelty pool, and more (PI 2026-10-10)
    assert {model_key(c) for c in novelty_pool() if is_plain(c)} < {model_key(c) for c in pool}


def test_the_live_pool_has_two_object_displays_and_redundant_features():
    pool = plain_pool()
    shapes = {c.shape for c in pool}
    assert {(2, 2), (2, 3), (2, 4), (4, 4)} <= shapes
    redundant = [c for c in pool if len(set(zip(*c.objects))) < len(c.objects[0])]
    assert redundant and any(c.shape == (2, 4) for c in redundant)
    # the PI's example: one object with three features, the other with those and a fourth
    assert any(sorted(map(sum, c.objects)) == [3, 4] and c.shape == (2, 4) for c in pool)
    # two words with one extension are one display: only the first is in the pool
    for c in pool:
        if c.utterance is not None:
            cols = list(zip(*c.objects))
            assert cols[c.utterance] not in cols[:c.utterance]
    assert len({model_key(c) for c in pool}) == len(pool)
    assert len(PLAIN_RSA_LENSES) == len(DEFAULT_RSA_LENSES) - 1
    assert not any("favorite" in lens for lens in PLAIN_RSA_LENSES)
