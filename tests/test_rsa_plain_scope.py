"""The live phase's scope: plain displays only (PI 2026-10-10)."""

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
    assert all(is_plain(c) for c in plain_pool()) and len(plain_pool()) < len(novelty_pool())
    assert len(PLAIN_RSA_LENSES) == len(DEFAULT_RSA_LENSES) - 1
    assert not any("favorite" in lens for lens in PLAIN_RSA_LENSES)
