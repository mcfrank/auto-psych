"""Unit tests for the pieces of ``scripts/paper/sr_figure_data.py``."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.paper import sr_figure_data as sr


def _row(a, b, chose_left):
    return {"sequence_a": a, "sequence_b": b, "chose_left": str(chose_left)}


def test_conditions_pool_both_orders_of_a_pair_onto_its_sorted_first_sequence():
    rows = [_row("HT", "TH", 1), _row("TH", "HT", 1), _row("TH", "HT", 0), _row("HH", "TT", 0)]
    p_left = np.array([0.8, 0.4, 0.4, 0.3])

    conditions = sr.condition_table(rows, {"m": p_left})

    by_pair = {tuple(c["pair"]): c for c in conditions}
    ht = by_pair[("HT", "TH")]
    # P(choose "HT"): trial 1 chose it on the left, trial 2 chose "TH" on the
    # left, trial 3 chose "HT" on the right.
    assert ht["n"] == 3
    assert ht["observed"] == pytest.approx(2 / 3)
    assert ht["m"] == pytest.approx((0.8 + 0.6 + 0.6) / 3)
    hh = by_pair[("HH", "TT")]
    assert hh["observed"] == 0.0  # chose the right one, "TT"
    assert hh["m"] == pytest.approx(0.3)


def test_conditions_refuse_predictions_that_do_not_cover_every_trial():
    with pytest.raises(ValueError, match="2 predictions for 1 trials"):
        sr.condition_table([_row("HT", "TH", 1)], {"m": np.array([0.5, 0.5])})


def _fake_compare(elpd, se_between):
    """compare_table restricted to ``names``: the lower model's row carries the
    clustered SE of its difference from the higher one."""

    def compare(responses, models_dir, *, names=None, **kwargs):
        names = list(names) if names is not None else list(elpd)
        top = max(names, key=elpd.get)
        table = {}
        for n in names:
            # Pairs the test does not name are unused by the code (the
            # all-model call reads only ELPDs); a NaN would show if read.
            se = 0.0 if n == top else se_between.get(frozenset((n, top)), float("nan"))
            table[n] = {
                "elpd_loo": elpd[n],
                "elpd_diff": elpd[top] - elpd[n],
                "dse_clustered": se,
                "loo_unreliable": False,
                "not_converged": False,
            }
        return table

    return compare


def test_every_model_is_compared_with_the_best_seed_pairwise():
    elpd = {"winner": -90.0, "s1": -100.0, "s2": -120.0}
    se = {frozenset(("winner", "s1")): 3.0, frozenset(("s2", "s1")): 5.0}

    scored = sr.score_against_best_seed(
        "responses.csv", "models", winner="winner", seeds=["s1", "s2"],
        compare=_fake_compare(elpd, se),
    )

    assert scored["best_seed"] == "s1"
    assert scored["elpd"] == elpd
    assert scored["vs_best_seed"] == {
        "winner": {"diff": 10.0, "se": 3.0},
        "s1": {"diff": 0.0, "se": 0.0},
        "s2": {"diff": -20.0, "se": 5.0},
    }


def test_an_unconverged_winner_or_best_seed_raises():
    elpd = {"winner": -90.0, "s1": -100.0}
    compare = _fake_compare(elpd, {frozenset(("winner", "s1")): 3.0})

    def broken(*args, **kwargs):
        table = compare(*args, **kwargs)
        table["s1"]["not_converged"] = True
        return table

    with pytest.raises(RuntimeError, match="s1"):
        sr.score_against_best_seed(
            "responses.csv", "models", winner="winner", seeds=["s1"], compare=broken
        )


# --- all runs pooled (Figure 3) ---------------------------------------------

_HEADER = "sequence_a,sequence_b,participant_id,trial_index,chose_left\n"


def test_pooled_responses_keep_each_runs_participants_apart(tmp_path):
    run1 = tmp_path / "run1.csv"
    run1.write_text(_HEADER + "HT,TH,0,1,1\nHH,TT,1,1,0\nTH,HT,0,2,0\n")
    run2 = tmp_path / "run2.csv"
    run2.write_text(_HEADER + "HT,TH,0,1,0\n")
    out = tmp_path / "pooled.csv"

    sr.pool_responses({"run1": run1, "run2": run2}, out)

    rows = sr._read_rows(out)
    # Run 1's participants 0 and 1, then run 2's participant 0 as a new one.
    assert [r["participant_id"] for r in rows] == ["0", "1", "0", "2"]
    assert [r["chose_left"] for r in rows] == ["1", "0", "0", "0"]
    assert list(rows[0]) == ["sequence_a", "sequence_b", "participant_id", "trial_index", "chose_left"]


def test_pooling_refuses_a_run_with_other_columns(tmp_path):
    run1 = tmp_path / "run1.csv"
    run1.write_text(_HEADER + "HT,TH,0,1,1\n")
    run2 = tmp_path / "run2.csv"
    run2.write_text("sequence_a,sequence_b,participant_id,trial_index,chose_left,extra\nHT,TH,0,1,1,x\n")
    with pytest.raises(ValueError, match="run2"):
        sr.pool_responses({"run1": run1, "run2": run2}, tmp_path / "pooled.csv")


def test_pooled_model_files_merge_the_runs_and_refuse_a_name_with_different_code(tmp_path):
    def model(name, text):
        path = tmp_path / f"{name}_{abs(hash(text))}.py"
        path.write_text(text)
        return path

    seed_1, seed_2 = model("s", "seed"), model("s", "seed")
    files = sr.pooled_model_files({
        "run1": {"w1": model("w1", "one"), "s": seed_1},
        "run2": {"w2": model("w2", "two"), "s": seed_2},
    })
    assert sorted(files) == ["s", "w1", "w2"]

    with pytest.raises(RuntimeError, match="'s'"):
        sr.pooled_model_files({"run1": {"s": seed_1}, "run2": {"s": model("s", "other seed")}})


def test_the_pooled_best_model_and_best_seed_have_the_highest_pooled_elpd():
    elpd = {"w1": -300.0, "w2": -290.0, "s1": -310.0, "s2": -305.0}
    pooled = sr.score_pooled(
        "pooled.csv", "models", winners={"w1": "run1", "w2": "run2"}, seeds=["s1", "s2"],
        compare=_fake_compare(elpd, {}),
    )
    assert pooled["best_model"] == "w2"
    assert pooled["best_seed"] == "s2"
    assert pooled["elpd"] == elpd
    assert pooled["winner_runs"] == {"w1": "run1", "w2": "run2"}


def test_an_unconverged_pooled_best_model_raises():
    compare = _fake_compare({"w1": -300.0, "s1": -310.0}, {})

    def broken(*args, **kwargs):
        table = compare(*args, **kwargs)
        table["w1"]["loo_unreliable"] = True
        return table

    with pytest.raises(RuntimeError, match="w1"):
        sr.score_pooled("pooled.csv", "models", winners={"w1": "run1"}, seeds=["s1"], compare=broken)


def test_an_unconverged_pooled_winner_raises_even_when_another_is_best():
    # Its ELPD-LOO is not a measurement, so "the best of the winners" is not
    # known: a well-fitted copy of it might have won.
    compare = _fake_compare({"w1": -300.0, "w2": -400.0, "s1": -310.0}, {})

    def broken(*args, **kwargs):
        table = compare(*args, **kwargs)
        table["w2"]["not_converged"] = True
        return table

    with pytest.raises(RuntimeError, match="w2"):
        sr.score_pooled(
            "pooled.csv", "models", winners={"w1": "run1", "w2": "run2"}, seeds=["s1"],
            compare=broken,
        )


def test_an_unconverged_seed_raises_even_when_another_seed_is_best():
    elpd = {"winner": -90.0, "s1": -100.0, "s2": -120.0}
    compare = _fake_compare(elpd, {frozenset(("winner", "s1")): 3.0})

    def broken(*args, **kwargs):
        table = compare(*args, **kwargs)
        table["s2"]["loo_unreliable"] = True
        return table

    with pytest.raises(RuntimeError, match="s2"):
        sr.score_against_best_seed(
            "responses.csv", "models", winner="winner", seeds=["s1", "s2"], compare=broken
        )


# --- the other models live at a run's end (Figure 2) ------------------------


def _posterior(**overrides):
    return {"best_model": "w", "n_trials": 100, "excluded_unreliable": [],
            "elpd_loo": {"w": -80.0, "r1": -85.0, "r2": -90.0}, **overrides}


def _record():
    return {"run": "run1", "n_trials": 100, "best_model": "w", "best_seed": "s1",
            "elpd": {"w": -80.5, "s1": -100.0}}


def test_the_other_models_at_a_runs_end_are_measured_from_its_best_seed():
    assert sr.others_at_end(_posterior(), _record()) == {"r1": 15.0, "r2": 10.0}


def test_the_loops_scores_on_other_data_raise():
    with pytest.raises(RuntimeError, match="99 trials"):
        sr.others_at_end(_posterior(n_trials=99), _record())


def test_a_model_the_loop_could_not_score_raises():
    with pytest.raises(RuntimeError, match="r2"):
        sr.others_at_end(_posterior(excluded_unreliable=["r2"]), _record())


def test_a_loop_record_without_the_winner_raises():
    posterior = _posterior(elpd_loo={"r1": -85.0})
    with pytest.raises(RuntimeError, match="'w'"):
        sr.others_at_end(posterior, _record())


# --- an accepted convergence failure (pooled only) --------------------------


def _pooled_with(problems, accepted):
    compare = _fake_compare({"w1": -300.0, "w2": -290.0, "s1": -310.0},
                            {frozenset(("w1", "s1")): 5.0, frozenset(("w2", "s1")): 6.0})

    def flagged(*args, **kwargs):
        table = compare(*args, **kwargs)
        for problem in problems:
            if "w2" in table:  # the pairwise comparisons without it are clean
                table["w2"][problem] = True
        return table

    return sr.score_pooled("pooled.csv", "models", winners={"w1": "run1", "w2": "run2"},
                           seeds=["s1"], compare=flagged, accepted=accepted)


def test_an_accepted_convergence_failure_is_scored_and_recorded():
    pooled = _pooled_with(["not_converged"], {"w2": "one person's two-humped posterior"})
    assert pooled["best_model"] == "w2"
    # Figure 2's pooled view: each against the best seed, with its paired SE.
    assert pooled["vs_best_seed"] == {"w1": {"diff": 10.0, "se": 5.0}, "w2": {"diff": 20.0, "se": 6.0},
                                      "s1": {"diff": 0.0, "se": 0.0}}
    assert pooled["accepted_fit_problems"] == {
        "w2": {"problems": ["not converged"], "reason": "one person's two-humped posterior"}}


def test_an_accepted_model_with_an_unreliable_psis_loo_still_raises():
    with pytest.raises(RuntimeError, match="w2: unreliable PSIS-LOO"):
        _pooled_with(["not_converged", "loo_unreliable"], {"w2": "reason"})


def test_an_acceptance_for_a_model_that_converged_raises_as_stale():
    with pytest.raises(RuntimeError, match="'w2'.*converged"):
        _pooled_with([], {"w2": "reason"})


def test_an_acceptance_for_a_model_not_compared_raises_as_stale():
    with pytest.raises(RuntimeError, match="'w9'"):
        _pooled_with(["not_converged"], {"w2": "reason", "w9": "reason"})
