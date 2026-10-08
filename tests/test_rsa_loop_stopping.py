"""The loop stops early once rounds stop changing its best model (PI decision 2026-10-08).

In Sherlock run 2 no real cell improved at its last round, so the rule saves
rounds; but a round without a new best model is often followed by the run's
biggest gain (real_rep1: unchanged at round 6, +40 lpd held out at round 7), so
it waits for two such rounds in a row.
"""

from types import SimpleNamespace

from src.rsa.loop.orchestrator import RSALoop, stale_rounds


def step(round_, best):
    return dict(round=round_, best_model=best)


def test_stale_rounds_counts_the_trailing_rounds_without_a_new_best_model():
    assert stale_rounds([step(-1, "rsa_l2")]) == 0
    assert stale_rounds([step(-1, "rsa_l2"), step(0, "rsa_l2")]) == 1  # the seeds' best kept
    assert stale_rounds([step(-1, "a"), step(0, "b"), step(1, "b"), step(2, "b")]) == 2
    assert stale_rounds([step(-1, "a"), step(0, "a"), step(1, "b")]) == 0


def fake_loop(max_iterations, stop_after, bests):
    """A loop whose round r makes bests[r] the best model."""
    loop = RSALoop.__new__(RSALoop)
    loop.cfg = SimpleNamespace(max_iterations=max_iterations, stop_after_stale_rounds=stop_after)
    loop.history = [step(-1, "seed")]
    loop.ran, loop.live = [], {}
    loop.setup = lambda: None
    loop.score = lambda r, events: None
    loop.run_round = lambda r: (loop.ran.append(r), loop.history.append(step(r, bests[r])))
    loop.end = lambda stopped=None: dict(stopped=stopped, rounds=list(loop.ran))
    return loop


def test_two_rounds_without_a_new_best_model_end_the_run():
    out = fake_loop(8, 2, ["a", "a", "b", "b", "b", "c", "d", "e"]).run()
    assert out["rounds"] == [0, 1, 2, 3, 4] and "2 rounds" in out["stopped"]


def test_one_unchanged_round_does_not_stop_it_and_0_never_stops():
    # Run 2's real_rep1: rsa_l2 stayed best after round 1, then the loop found better models.
    assert fake_loop(4, 2, ["seed", "a", "a", "b"]).run()["rounds"] == [0, 1, 2, 3]
    assert fake_loop(4, 0, ["seed", "seed", "seed", "seed"]).run()["rounds"] == [0, 1, 2, 3]
