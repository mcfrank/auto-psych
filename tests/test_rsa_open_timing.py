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
    from src.rsa_open.timing import DEFAULT_RECORDS, estimate

    p = paradigm_of("choice")
    lists = trial_lists(p, units_of(p), seed=0, n_lists=5)
    t = estimate(lists, DEFAULT_RECORDS)
    assert estimated_minutes(lists) == pytest.approx(t["estimated_minutes"])
    assert t["estimated_minutes"] == pytest.approx(t["model_minutes"] * t["calibration"])
    assert estimated_minutes(lists, []) == pytest.approx(t["model_minutes"])


def test_run_d2_s_records_are_its_committed_timing_files():
    import json
    from pathlib import Path

    from src.rsa_open.timing import CLOSED_CAMPAIGN, DEFAULT_RECORDS, OPEN_PAGE_SMOKE, TimingRecord

    smoke = Path(__file__).resolve().parents[1] / "data" / "rsa_open" / "smoke"
    for record, arm in zip(OPEN_PAGE_SMOKE, ("open", "fixed")):
        (stored,) = json.loads((smoke / f"{arm}_chain_0_smoke" / "experiment1" / "data" / "timing.json").read_text())
        stored["source"] = record.source
        assert TimingRecord.from_json(stored) == record
    assert DEFAULT_RECORDS == CLOSED_CAMPAIGN + OPEN_PAGE_SMOKE


def test_this_campaign_s_page_calibrates_before_the_closed_campaign_s():
    """PI 2026-10-29: run D2's people took about 1.3 times the closed-calibrated estimate on both arms."""
    from src.rsa_open.timing import CLOSED_CAMPAIGN, DEFAULT_RECORDS, OPEN_PAGE_SMOKE, calibration

    k_closed, _ = calibration(CLOSED_CAMPAIGN, "choice")
    k_choice, basis = calibration(DEFAULT_RECORDS, "choice")
    assert basis == "1 choice experiment(s) on this campaign's page" and 1.15 < k_choice / k_closed < 1.4
    k_bet, basis = calibration(DEFAULT_RECORDS, "bet")
    assert basis == "1 bet experiment(s) on this campaign's page" and k_bet > k_closed
    k_slider, basis = calibration(DEFAULT_RECORDS, "slider")
    assert basis == "2 experiment(s) of other response types on this campaign's page (none of slider yet)"
    assert min(k_choice, k_bet) <= k_slider <= max(k_choice, k_bet)
    assert calibration(OPEN_PAGE_SMOKE, "choice") == (k_choice, "1 choice experiment(s) on this campaign's page")


def test_the_pay_rate_is_taken_at_the_larger_of_declared_and_estimated_minutes():
    from src.rsa_open.timing import hourly_usd

    assert hourly_usd(1.0, 6.0, 7.85) == pytest.approx(60 / 7.85)  # run D2's open study: $7.64 an hour
    assert hourly_usd(0.5, 3.0, 2.9) == pytest.approx(10.0)
