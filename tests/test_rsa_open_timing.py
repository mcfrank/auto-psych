"""Completion times (src/rsa_open/timing.py): the closed campaign's records, the calibrated estimate
the tools and the pilot share, and the designer's digest."""

import pytest

from tests.test_rsa_open_screens import paradigm_of, units_of


def test_the_closed_records_are_the_campaign_reports_medians():
    from src.rsa_open.timing import CLOSED_CAMPAIGN

    assert len(CLOSED_CAMPAIGN) == 9
    assert {r.response_type for r in CLOSED_CAMPAIGN} == {"choice"}
    assert sorted(r.median_s for r in CLOSED_CAMPAIGN)[4] == 126  # the median experiment's median


def test_calibrated_on_the_closed_campaign_the_model_predicts_its_people():
    from src.rsa_open.timing import CLOSED_CAMPAIGN, calibration, model_seconds

    k, basis = calibration(CLOSED_CAMPAIGN, "choice")
    assert 0.3 < k < 0.6 and basis.startswith("9 choice")  # the raw model was about twice too slow
    for r in CLOSED_CAMPAIGN:
        predicted = k * model_seconds(r.response_type, r.materials, r.trials, r.objects_per_trial,
                                      r.words_per_trial, r.instruction_words)
        assert r.p10_s < predicted < r.p90_s


def test_a_response_type_with_records_of_its_own_is_calibrated_on_them():
    from src.rsa_open.timing import CLOSED_CAMPAIGN, TimingRecord, calibration

    k_other, basis = calibration(CLOSED_CAMPAIGN, "production")
    assert "none of production yet" in basis
    slow = TimingRecord("open c1 e1", "production", "text", 27, 3.4, 50.0, 160, 100, 600, 300, 900)
    k_own, basis = calibration([*CLOSED_CAMPAIGN, slow], "production")
    assert basis.startswith("1 production") and k_own > k_other
    assert calibration([], "choice") == (1.0, "no timing records: the model's estimate, uncalibrated")


def test_a_finished_experiment_becomes_a_record_and_a_digest_row():
    from src.rsa_open.screens import trial_lists
    from src.rsa_open.timing import digest_md, record_from_sessions

    p = paradigm_of("rating")
    lists = trial_lists(p, units_of(p), seed=0, n_lists=10)
    record = record_from_sessions("open chain 1, experiment 1", lists, [100, 120, 140, 160, 600])
    assert (record.response_type, record.materials, record.trials) == ("rating", "svg", 9)
    assert record.median_s == 140 and record.n_people == 5
    assert "| open chain 1, experiment 1 | rating | svg | 9 |" in digest_md([record])
    with pytest.raises(ValueError, match="no completion times"):
        record_from_sessions("empty", lists, [])


def test_the_pilot_s_estimate_is_the_calibrated_one():
    from src.rsa_open.pilot import estimated_minutes
    from src.rsa_open.screens import trial_lists
    from src.rsa_open.timing import CLOSED_CAMPAIGN, estimate

    p = paradigm_of("choice")
    lists = trial_lists(p, units_of(p), seed=0, n_lists=5)
    t = estimate(lists, CLOSED_CAMPAIGN)
    assert estimated_minutes(lists) == pytest.approx(t["estimated_minutes"])
    assert t["estimated_minutes"] == pytest.approx(t["model_minutes"] * t["calibration"])
    assert estimated_minutes(lists, []) == pytest.approx(t["model_minutes"])
