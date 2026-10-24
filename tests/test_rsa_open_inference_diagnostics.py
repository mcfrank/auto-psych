"""The open loop's inference fixes after run B (PI 2026-10-23, docs/auto_open/PLAN.md):

- every fit records its steps per draw, step size and timing, and `src.rsa_open.sampler` names
  what they mean (correlated parameter pairs, many steps per draw) in the self-check and in a
  rejection's reason;
- the self-check projects the admission fit's time on all the data and fails over the limit;
- a fit that fails the convergence gate is refitted only when it is a near miss (main's rule);
- a theory's cross-validation folds are bounded at a few times its own full fit, and a failed
  fold leaves it untrusted with the reason on record;
- the theorist's prompt no longer invites a listener-sharpness parameter and names RSA's
  identification pairs.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.rsa.fit import FitSettings
from tests.test_rsa_open_loop_fitting import QUICK, corpus, theory_path  # noqa: F401 (fixtures)

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- the report

def test_a_ridge_is_named_and_independent_parameters_are_not():
    from src.rsa_open.sampler import correlated_pairs

    rng = np.random.default_rng(0)
    alpha = rng.lognormal(0, 0.5, (4, 500))
    samples = {"alpha": alpha, "cost_weight": 2.0 / alpha * rng.lognormal(0, 0.05, (4, 500)),
               "lapse": rng.beta(2, 20, (4, 500)), "fixed": np.ones((4, 500))}
    pairs = correlated_pairs(samples, ["alpha", "cost_weight", "lapse", "fixed"])
    assert [(a, b) for a, b, _ in pairs] == [("alpha", "cost_weight")] and pairs[0][2] < -0.85


def test_a_vector_parameter_is_reported_by_entry():
    from src.rsa_open.sampler import correlated_pairs

    rng = np.random.default_rng(1)
    w = rng.normal(size=(2, 300))
    samples = {"w": np.stack([w, w + 0.01 * rng.normal(size=(2, 300)), rng.normal(size=(2, 300))], axis=-1)}
    assert [(a, b) for a, b, _ in correlated_pairs(samples, ["w"])] == [("w[0]", "w[1]")]


def test_steps_per_draw_counts_the_slowest_chain_and_the_report_says_what_many_steps_mean():
    from src.rsa_open.sampler import SamplerReport, steps_per_draw

    assert steps_per_draw(np.array([[3, 7, 3], [15, 7, 7]])) == pytest.approx(29 / 3)
    many = SamplerReport(13.0, 0.28, (("alpha", "cost_weight", 0.92),))
    text = many.text()
    assert "13.0 leapfrog steps per draw" in text and "narrow or curved" in text
    assert "alpha and cost_weight are correlated at +0.92" in text
    assert "narrow" not in SamplerReport(6.8, 0.66, ()).text()
    assert SamplerReport.from_json(json.loads(json.dumps(many.to_json()))) == many


def test_the_projection_takes_the_short_fit_s_steps_to_the_full_fit_at_the_full_data_s_step_cost():
    from src.rsa_open.sampler import projected_fit_seconds

    check = FitSettings(num_warmup=300, num_samples=300, num_chains=2)
    full = FitSettings(num_warmup=1000, num_samples=1000, num_chains=4)
    timing = dict(warmup_sec=50.0, sampling_sec=15.0, warmup_steps=3000.0, sampling_steps=2000.0)
    # The short warmup's 3,000 steps, then 700 more warmup and 1,000 sampling iterations at its 6.67 steps per draw.
    expected = 0.010 * (3000 + (700 + 1000) * 2000 / 300)
    assert projected_fit_seconds(timing, check, full, 0.010) == pytest.approx(expected)


def test_a_step_costs_one_gradient_for_every_chain(theory_path, corpus):  # noqa: F811
    from src.rsa_open.fit import TheoryModel
    from src.rsa_open.sampler import step_seconds

    blocks, _ = corpus
    theory = TheoryModel(theory_path)
    one, four = step_seconds(theory, blocks, 1, repeats=5), step_seconds(theory, blocks, 4, repeats=5)
    assert 0 < one < 1 and 0 < four < 1


# --------------------------------------------------------------------------- the fits record it

def test_a_fit_records_its_steps_step_size_and_timing_through_the_cache(theory_path, corpus, tmp_path):  # noqa: F811
    from src.rsa_open.loop.fitting import _read, fingerprint, fit_cached
    from src.rsa.loop.fitting import cache_paths

    _, path = corpus
    fitted = fit_cached(theory_path, "rsa_l1", path, QUICK, tmp_path)
    assert {"warmup_sec", "sampling_sec", "warmup_steps", "sampling_steps"} == set(fitted.timing)
    assert fitted.idata.sample_stats["n_steps"].shape == (1, 40)
    report = fitted.sampler_report()
    assert report is not None and report.steps_per_draw >= 1 and report.step_size > 0
    again = _read(*cache_paths(tmp_path, "rsa_l1", fingerprint(theory_path, path, QUICK)))
    assert again.timing == fitted.timing and again.sampler_report() == report


def test_a_fit_killed_while_sampling_is_rejected_with_its_warmup_report(tmp_path):
    from src.rsa_open.loop.fitting import _warmup_note, warmup_path
    from src.rsa_open.sampler import SamplerReport

    nc = tmp_path / "t.abc.nc"
    assert "still in warmup" in _warmup_note(nc)
    report = SamplerReport(14.0, 0.2, (("alpha", "beta", -0.95),), "warmup")
    warmup_path(nc).write_text(json.dumps(dict(report=report.to_json(), timing=dict(warmup_sec=1200.0))))
    note = _warmup_note(nc)
    assert "warmup took 20 minutes" in note and "by the end of warmup" in note and "alpha and beta" in note


def test_the_child_writes_its_warmup_report_beside_the_fit(theory_path, corpus, tmp_path):  # noqa: F811
    from src.rsa_open.loop.fitting import fingerprint, fit_cached, warmup_path
    from src.rsa.loop.fitting import cache_paths

    _, path = corpus
    fit_cached(theory_path, "rsa_l1", path, QUICK, tmp_path, time_limit_sec=600)
    nc, _ = cache_paths(tmp_path, "rsa_l1", fingerprint(theory_path, path, QUICK))
    record = json.loads(warmup_path(nc).read_text())
    assert record["report"]["phase"] == "warmup" and record["timing"]["warmup_sec"] > 0


# --------------------------------------------------------------------------- near-miss refits

def _fake_fit(converged: bool, near: bool):
    return SimpleNamespace(converged=converged, near_miss=near and not converged)


@pytest.mark.parametrize("first, refits", [(_fake_fit(True, False), 0), (_fake_fit(False, True), 1),
                                           (_fake_fit(False, False), 0)])
def test_only_a_near_miss_is_refitted(monkeypatch, first, refits, tmp_path):
    import src.rsa_open.loop.fitting as fitting

    calls = []

    def fake(theory_path, name, corpus_path, settings, cache_dir, **kw):
        calls.append(settings)
        return first if len(calls) == 1 else _fake_fit(True, False)

    monkeypatch.setattr(fitting, "fit_cached", fake)
    out = fitting.loop_fit(tmp_path / "t.py", "t", tmp_path / "c.json", FitSettings(), tmp_path)
    assert len(calls) == 1 + refits
    assert out is (first if not refits else out) and (refits == 0 or calls[1].target_accept == 0.95)


def test_the_near_miss_rule_is_main_s():
    from src.rsa_open.corpus import CorpusFit
    import arviz as az

    rng = np.random.default_rng(0)

    def fit(alpha, diverging):
        idata = az.from_dict(posterior={"alpha": alpha}, sample_stats={"diverging": diverging})
        from src.models.pymc_inference import convergence_problems
        return CorpusFit("t", ["E"], idata, ["alpha"], convergence_problems(idata, ["alpha"]))

    good = rng.normal(size=(4, 500))
    assert fit(good, np.zeros((4, 500), bool)).converged
    few_divergent = np.zeros((4, 500), bool)
    few_divergent[0, :10] = True  # 0.5%: fails the 0.1% gate, within the 2% near miss
    assert fit(good, few_divergent).near_miss
    stuck = good + np.array([0, 0, 0, 5.0])[:, None]  # one chain elsewhere: R-hat far above 1.2
    assert not fit(stuck, np.zeros((4, 500), bool)).near_miss


# --------------------------------------------------------------------------- fold bounds

def _loop(tmp_path, timing):
    from src.rsa_open.loop.orchestrator import OpenLoop, OpenLoopConfig

    loop = OpenLoop.__new__(OpenLoop)
    loop.cfg = OpenLoopConfig(corpus_path=tmp_path, seed_models_dir=tmp_path, results_dir=tmp_path, cv_folds=2)
    loop.live_cfg = SimpleNamespace(time_limit_sec=float("inf"))
    fit = SimpleNamespace(timing=timing, flat=lambda: {"alpha": np.ones(10)}, converged=True)
    loop.live = {"slow": SimpleNamespace(theory=None, fit=fit)}
    loop.models_dir, loop.dir, loop.fold_paths = tmp_path, tmp_path, [tmp_path / "f0", tmp_path / "f1"]
    loop.cv, loop.cv_problems, loop.admission_limit_sec = {}, {}, None
    loop.blocks = [SimpleNamespace(name="E")]
    return loop


def test_a_fold_is_bounded_at_a_few_times_its_theory_s_full_fit(tmp_path):
    from src.rsa.loop.fitting import FIT_TIME_LIMIT_SEC
    from src.rsa_open.loop.orchestrator import FOLD_MIN_SEC, FOLD_TIME_FACTOR

    assert _loop(tmp_path, dict(warmup_sec=1800.0, sampling_sec=1200.0)).fold_time_limit("slow") == FOLD_TIME_FACTOR * 3000
    assert _loop(tmp_path, dict(warmup_sec=60.0, sampling_sec=30.0)).fold_time_limit("slow") == FOLD_MIN_SEC
    assert _loop(tmp_path, {}).fold_time_limit("slow") == FOLD_TIME_FACTOR * FIT_TIME_LIMIT_SEC
    unlimited = _loop(tmp_path, {})
    unlimited.live_cfg = SimpleNamespace(time_limit_sec=None)  # in-process fits (tests): nothing is limited
    assert unlimited.fold_time_limit("slow") is None


def test_a_fold_that_fails_leaves_its_theory_unscored_and_untrusted_on_record(tmp_path, monkeypatch):
    import src.rsa_open.loop.orchestrator as orch
    from src.rsa.loop.fitting import FitTimeLimitExceeded

    limits = []

    def stuck(path, name, corpus, settings, cache, time_limit_sec=None, init=None):
        limits.append(time_limit_sec)
        raise FitTimeLimitExceeded(f"{name}: the fit did not finish within 45 minutes")

    monkeypatch.setattr(orch, "loop_fit", stuck)
    loop = _loop(tmp_path, dict(warmup_sec=600.0, sampling_sec=300.0))
    s = loop.standing()
    assert set(limits) == {orch.FOLD_TIME_FACTOR * 900}
    d = s["slow"]
    assert not d["trusted"] and not d["cv_converged"] and d["sel_elpd"] == -float("inf")
    assert "a cross-validation fold failed" in d["cv_problem"] and "45 minutes" in d["cv_problem"]
    assert loop._describe("slow", s) == d["cv_problem"]


# --------------------------------------------------------------------------- the self-check and the prompt

def test_the_self_check_fails_a_theory_projected_over_the_admission_limit(monkeypatch, tmp_path):
    import src.rsa_open.loop.check_candidate as cc
    from src.rsa_open.loop.gates import Admission
    from src.rsa_open.sampler import SamplerReport

    def fit(per_step_ms):
        return SimpleNamespace(timing=dict(warmup_sec=60.0, sampling_sec=per_step_ms * 2.0, warmup_steps=4000.0,
                                           sampling_steps=2000.0),
                               sampler_report=lambda: SamplerReport(13.0, 0.28, (("alpha", "cost_weight", 0.92),)))

    monkeypatch.setattr(cc, "read_corpus", lambda p: [])
    monkeypatch.setattr(cc, "TheoryModel", lambda *a, **k: None)
    monkeypatch.setattr(cc, "step_seconds", lambda theory, blocks, chains: per_step_ms / 1000)
    # 15,333 projected steps: 2 min at 8 ms a step, 51 min at 200 ms, judged against the loop's own limit.
    for per_step_ms, limit_min, ok in ((8.0, 30, True), (200.0, 30, False), (200.0, 90, True)):
        monkeypatch.setattr(cc, "admit", lambda *a, **k: Admission(True, "admitted", fit=fit(per_step_ms)))
        passed, message = cc.check(tmp_path, tmp_path / "check.json", tmp_path / "corpus.json", limit_min * 60)
        assert passed is ok, message
        assert "alpha and cost_weight are correlated at +0.92" in message
        assert "Projected admission fit on all the data" in message and "15,333 leapfrog steps" in message
        assert f"The loop's limit is {limit_min} minutes" in message
        assert (f"over the loop's {limit_min}-minute limit" in message) is (not ok)


def test_the_brief_states_the_loop_s_admission_limit_and_passes_it_to_the_self_check(tmp_path):
    from src.rsa_open.loop.brief import context_md

    def ctx(limit):
        return context_md(candidate_dir=tmp_path / "c", corpus_path=tmp_path / "check.json",
                          full_corpus_path=tmp_path / "corpus.json", admission_limit_sec=limit, blocks=[],
                          data_paths={}, paradigm_notes={}, round_index=0, n_rounds=3)

    text = ctx(90 * 60)
    assert "must finish within **90 minutes**" in text
    assert f"--full-corpus {tmp_path / 'corpus.json'} --limit-minutes 90`" in text
    assert "## The admission limit" not in ctx(None) and "--limit-minutes inf`" in ctx(None)


# --------------------------------------------------------------------------- the corpus-relative limit

def _limit_loop(tmp_path, minutes):
    from src.rsa_open.loop.gates import GateConfig

    loop = _loop(tmp_path, {})
    loop.live = {f"t{i}": SimpleNamespace(fit=SimpleNamespace(timing=dict(warmup_sec=60.0 * m * 0.6,
                                                                          sampling_sec=60.0 * m * 0.4) if m else {}))
                 for i, m in enumerate(minutes)}
    loop.gate_cfg = GateConfig(tmp_path, tmp_path, FitSettings(), 0.002, loop.cfg.fit_time_limit_sec)
    return loop


def test_the_admission_limit_is_twice_the_median_starting_fit_and_never_below_the_floor(tmp_path):
    from src.rsa.loop.fitting import FIT_TIME_LIMIT_SEC
    from src.rsa_open.loop.orchestrator import ADMISSION_FACTOR

    loop = _limit_loop(tmp_path, [13, 26, 41, 46, 83, 165])  # chain 0's starting theories on all of E (the audit)
    loop._set_admission_limit()
    assert loop.admission_limit_sec == ADMISSION_FACTOR * 43.5 * 60 == loop.gate_cfg.time_limit_sec
    small = _limit_loop(tmp_path, [1, 2, 3, None])  # fast theories (a fit recorded without its time is skipped)
    small._set_admission_limit()
    assert small.admission_limit_sec == FIT_TIME_LIMIT_SEC == small.gate_cfg.time_limit_sec


def test_the_admission_limit_refuses_starting_fits_without_their_time(tmp_path):
    loop = _limit_loop(tmp_path, [None, None])
    with pytest.raises(RuntimeError, match="fresh work root"):
        loop._set_admission_limit()


def test_an_unlimited_loop_has_no_admission_limit(tmp_path):
    loop = _limit_loop(tmp_path, [None])
    loop.cfg.fit_time_limit_sec = None
    loop._set_admission_limit()
    assert loop.admission_limit_sec is None


def test_a_replay_reruns_only_the_loops_of_an_earlier_rehearsal_s_experiments():
    text = (ROOT / "scripts/rsa/slurm/open_rehearsal.sbatch").read_text()
    assert "REPLAY_FROM" in text and "--exclude '/loop/'" in text and "--exclude '/loop.abandoned_*/'" in text
    assert "this WORK_ROOT's ground truth is not REPLAY_FROM's" in text
    # The source path names another rehearsal's root (its private/ holds the truth): never the agents'.
    assert text.index("unset REPLAY_FROM") < text.index("WITHHELD=()")


def test_the_prompt_names_the_identification_pairs_and_no_longer_invites_a_sharpness_parameter():
    text = " ".join((ROOT / "src/rsa_open/loop/prompts/open_theory.md").read_text().split())
    assert "e.g. `beta`" not in text and "put it in the theory as a parameter of its own" not in text
    for phrase in ("**rationality and cost**", "**rationality and listener sharpness**", "**prior weights and lapse**",
                   "correlated above 0.85", "says what in the data separates them", "Do not add a parameter"):
        assert phrase in text, phrase
