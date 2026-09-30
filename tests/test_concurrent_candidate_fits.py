"""A wave's candidates are fitted concurrently, then admitted in slot order.

Admission used to fit one candidate at a time (4 of a cell's 16 CPUs). Now
``prefit_candidates`` fits every candidate of a wave that passes the cheap
gates concurrently, into the fit cache, before the sequential admission,
which then loads those fits. Admission itself is unchanged, so its verdicts
must be exactly those of sequential admission.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import src.pipelines.inner_loop.model_zoo as model_zoo
import src.pipelines.inner_loop.pymc_orchestrator as pymc_orchestrator
from src.models import pymc_inference as pi
from src.pipelines.inner_loop.hypothesis_ledger import HypothesisLedger
from tests.paths import PYMC_MODEL_FIXTURES_DIR
from tests.test_parallel_candidates import _patch_loop_internals
from tests.inner_loop_fixtures import write_responses, write_seed_models


@pytest.fixture(autouse=True)
def _clean_fit_cache():
    pi.clear_fit_cache()
    yield
    pi.clear_fit_cache()


# ---------------------------------------------------------------------------
# Which candidates are fitted, and when
# ---------------------------------------------------------------------------


def _candidate(
    root: Path,
    label: str,
    *,
    source="# candidate\n",
    hypothesis="People do H.\n",
    name=None,
):
    directory = root / label
    directory.mkdir(parents=True)
    (directory / "candidate.py").write_text(source, encoding="utf-8")
    if hypothesis is not None:
        (directory / "hypothesis.md").write_text(hypothesis, encoding="utf-8")
    if name is not None:
        (directory / "model_name.txt").write_text(name, encoding="utf-8")
    return directory / "candidate.py"


def test_only_candidates_that_pass_the_cheap_gates_are_fitted(tmp_path, monkeypatch):
    fitted = {}

    def fake_concurrent(names, models_dir, responses_path, **kwargs):
        fitted["names"] = list(names)
        fitted["dir"] = Path(models_dir)
        fitted["staged"] = sorted(p.name for p in Path(models_dir).glob("*.py"))
        fitted["kwargs"] = kwargs

    monkeypatch.setattr(model_zoo, "fit_time_limited_concurrently", fake_concurrent)
    monkeypatch.setattr(model_zoo, "load_pymc_model", lambda name, d: object())
    monkeypatch.setattr(
        model_zoo,
        "model_logp_is_finite",
        lambda name, d, r: (name != "nan_logp", "logp is NaN"),
    )
    monkeypatch.setattr(model_zoo, "model_contract_violation", lambda *a, **k: None)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    candidates = [
        (_candidate(tmp_path, "c0"), "good"),
        (_candidate(tmp_path, "c1", hypothesis=None), "no_hypothesis"),
        (_candidate(tmp_path, "c2", source="import os\n"), "forbidden"),
        (_candidate(tmp_path, "c3"), "nan_logp"),
        (tmp_path / "c4" / "candidate.py", "never_written"),
        (_candidate(tmp_path, "c5"), "also_good"),
    ]

    sent = model_zoo.prefit_candidates(
        candidates,
        tmp_path / "r.csv",
        cache_dir=tmp_path / "cache",
        fit_kwargs={"draws": 7},
    )

    assert sent == fitted["names"] == ["good", "also_good"]
    assert fitted["staged"] == ["also_good.py", "good.py"]
    assert (
        fitted["dir"] != models_dir and not fitted["dir"].exists()
    )  # scratch, removed
    assert fitted["kwargs"]["cache_dir"] == tmp_path / "cache"
    assert fitted["kwargs"]["fit_kwargs"] == {"draws": 7}
    assert fitted["kwargs"]["time_limit_sec"] == model_zoo.CANDIDATE_FIT_TIME_LIMIT_SEC
    assert list(models_dir.iterdir()) == []  # the zoo is untouched


def test_a_wave_is_prefitted_under_its_admission_names_before_any_admission(
    tmp_path, monkeypatch
):
    _patch_loop_internals(monkeypatch)
    events = []

    def spawn(candidate_dir, docs, **kwargs):
        (candidate_dir / "candidate.py").write_text("# candidate\n", encoding="utf-8")
        (candidate_dir / "hypothesis.md").write_text(
            "People use H.\n", encoding="utf-8"
        )
        if candidate_dir.name in ("candidate_0", "candidate_2"):
            (candidate_dir / "model_name.txt").write_text("same_idea", encoding="utf-8")
        return True

    def fake_prefit(candidates, responses_path, *, cache_dir, fit_kwargs):
        events.append(("prefit", [name for _, name in candidates], cache_dir))
        return [name for _, name in candidates]

    real_admit = pymc_orchestrator._admit_candidate_with_reason

    def recording_admit(candidate_file, models_dir, model_name, *a, **k):
        events.append(("admit", model_name))
        return real_admit(candidate_file, models_dir, model_name, *a, **k)

    monkeypatch.setattr(pymc_orchestrator, "_spawn_candidate_agent", spawn)
    monkeypatch.setattr(pymc_orchestrator, "prefit_candidates", fake_prefit)
    monkeypatch.setattr(
        pymc_orchestrator, "_admit_candidate_with_reason", recording_admit
    )
    pymc_orchestrator.run_pymc_inner_loop(
        responses_path=write_responses(tmp_path),
        results_dir=tmp_path / "model_loop",
        seed_models_dir=write_seed_models(tmp_path, ["model_a"]),
        max_iterations=1,
        candidate_count=3,
        enable_critique=False,
        fit_kwargs={},
        cache_dir=tmp_path / "cache",
    )

    predicted = ["same_idea", "iter0_candidate1", "same_idea_2"]
    assert events[0] == ("prefit", predicted, tmp_path / "cache")
    assert events[1:] == [("admit", name) for name in predicted]


def test_without_a_fit_cache_there_is_no_prefit(tmp_path, monkeypatch):
    _patch_loop_internals(monkeypatch)

    def spawn(candidate_dir, docs, **kwargs):
        (candidate_dir / "candidate.py").write_text("# candidate\n", encoding="utf-8")
        (candidate_dir / "hypothesis.md").write_text(
            "People use H.\n", encoding="utf-8"
        )
        return True

    def no_prefit(*a, **k):
        raise AssertionError("prefit needs a cache dir to hand fits to admission")

    monkeypatch.setattr(pymc_orchestrator, "_spawn_candidate_agent", spawn)
    monkeypatch.setattr(pymc_orchestrator, "prefit_candidates", no_prefit)
    pymc_orchestrator.run_pymc_inner_loop(
        responses_path=write_responses(tmp_path),
        results_dir=tmp_path / "model_loop",
        seed_models_dir=write_seed_models(tmp_path, ["model_a"]),
        max_iterations=1,
        candidate_count=2,
        enable_critique=False,
        fit_kwargs={},
    )


def test_concurrent_fits_sample_first_fits_then_the_near_miss_refits(
    tmp_path, monkeypatch
):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    for name in ("fine", "near", "slow"):
        (models_dir / f"{name}.py").write_text(f"# {name}\n", encoding="utf-8")
    responses = tmp_path / "r.csv"
    responses.write_text("chose_left\n1\n", encoding="utf-8")
    batches = []

    refit_seeds = []

    def fake_sample(requests, *, time_limit_sec, workers):
        batches.append([(r.name, r.settings["target_accept"]) for r in requests])
        refit_seeds.extend(
            r.settings["random_seed"]
            for r in requests
            if r.settings["target_accept"] == 0.95
        )
        return [
            pi.FitTimeLimitExceeded(r.name, time_limit_sec, 0.8)
            if r.name == "slow"
            else None
            for r in requests
        ]

    monkeypatch.setattr(pi, "sample_fits_time_limited", fake_sample)
    monkeypatch.setattr(
        pi,
        "_fit_once",
        lambda name, *a: SimpleNamespace(name=name, fingerprint=f"fp-{name}"),
    )
    monkeypatch.setattr(
        pi, "_refit_decision", lambda name, fitted, settings: fitted.name == "near"
    )
    monkeypatch.setattr(pi, "allocated_cpus", lambda: 16)

    pi.fit_time_limited_concurrently(
        ["fine", "near", "slow"],
        models_dir,
        responses,
        cache_dir=tmp_path / "cache",
        fit_kwargs={"target_accept": 0.8, "chains": 4, "cores": 4},
        time_limit_sec=900,
    )

    assert batches == [[("fine", 0.8), ("near", 0.8), ("slow", 0.8)], [("near", 0.95)]]
    # The refit samples with its own seed, derived from the first fit's.
    assert refit_seeds == [pi.refit_random_seed(42, "fp-near")]


# ---------------------------------------------------------------------------
# Real fits: the verdicts of concurrent and of sequential admission agree
# ---------------------------------------------------------------------------

_REPRESENTATIVENESS = (PYMC_MODEL_FIXTURES_DIR / "representativeness.py").read_text()
_FAIR_COIN = (PYMC_MODEL_FIXTURES_DIR / "bayesian_fair_coin.py").read_text()


def _wave(root: Path):
    """Six candidates whose verdicts depend on their order: a model, its
    re-skin (a novelty rejection, but only after the first is admitted), a
    code-gate rejection, a missing hypothesis, a genuinely different model and
    a candidate that reuses the first one's name."""
    return [
        _candidate(root, "c0", source=_REPRESENTATIVENESS, name="rep_a"),
        _candidate(
            root, "c1", source="# a re-skin\n" + _REPRESENTATIVENESS, name="rep_b"
        ),
        _candidate(
            root, "c2", source="import os\n" + _REPRESENTATIVENESS, name="rep_c"
        ),
        _candidate(
            root, "c3", source=_REPRESENTATIVENESS, hypothesis=None, name="rep_d"
        ),
        _candidate(
            root,
            "c4",
            source=_REPRESENTATIVENESS.replace("sigma=5.0", "sigma=0.05"),
            name="rep_e",
        ),
        _candidate(root, "c5", source="# fair coin\n" + _FAIR_COIN, name="rep_a"),
    ]


def _admit_in_order(root: Path, *, prefit: bool, novelty_predictions=None):
    models_dir = root / "models"
    models_dir.mkdir(parents=True)
    shutil.copy(PYMC_MODEL_FIXTURES_DIR / "bayesian_fair_coin.py", models_dir)
    (models_dir / "models_manifest.yaml").write_text(
        yaml.safe_dump(
            {"models": [{"name": "bayesian_fair_coin", "rationale": "seed"}]}
        ),
        encoding="utf-8",
    )
    # The fixture responses, with the integer participant ids the novelty
    # gate marginalizes over.
    responses = root / "responses.csv"
    responses.write_text(
        (PYMC_MODEL_FIXTURES_DIR / "responses.csv").read_text().replace(",p", ","),
        encoding="utf-8",
    )
    cache_dir = root / "cache"
    fit_kwargs = {
        "draws": 300,
        "tune": 300,
        "chains": 2,
        "cores": 2,
        "target_accept": 0.8,
    }
    ledger = HypothesisLedger.create(
        root / "ledger.jsonl", inherit_from=root / "none.jsonl"
    )
    candidates = _wave(root / "wave")
    if prefit:
        claimed, predicted = [], []
        for i, candidate in enumerate(candidates):
            name = model_zoo._resolve_candidate_name(
                candidate.parent,
                models_dir,
                fallback=f"iter0_candidate{i}",
                taken=claimed,
                announce=False,
            )
            claimed.append(name)
            predicted.append((candidate, name))
        model_zoo.prefit_candidates(
            predicted, responses, cache_dir=cache_dir, fit_kwargs=fit_kwargs
        )
    sampled_before_admission = (
        sorted(p.name for p in cache_dir.glob("*.nc")) if prefit else None
    )
    verdicts = []
    for i, candidate in enumerate(candidates):
        name = model_zoo._resolve_candidate_name(
            candidate.parent, models_dir, fallback=f"iter0_candidate{i}"
        )
        admission = model_zoo._admit_candidate_with_reason(
            candidate,
            models_dir,
            name,
            responses,
            cache_dir=cache_dir,
            fit_kwargs=fit_kwargs,
            ledger=ledger,
            ledger_context=f"candidate {i}",
            **(
                {}
                if novelty_predictions is None
                else {"novelty_predictions": novelty_predictions}
            ),
        )
        verdicts.append((name, admission.admitted, admission.reason))
    pi.clear_fit_cache()
    return {
        "verdicts": verdicts,
        "manifest": (models_dir / "models_manifest.yaml").read_text(),
        "ledger": (root / "ledger.jsonl").read_text(),
        "fits": sorted(p.name for p in cache_dir.glob("*.nc")),
        "sampled_before_admission": sampled_before_admission,
    }


@pytest.mark.slow
def test_concurrent_and_sequential_admission_give_identical_verdicts(tmp_path):
    sequential = _admit_in_order(tmp_path / "sequential", prefit=False)
    concurrent = _admit_in_order(tmp_path / "concurrent", prefit=True)

    assert concurrent["verdicts"] == sequential["verdicts"]
    assert concurrent["manifest"] == sequential["manifest"]
    assert concurrent["ledger"] == sequential["ledger"]
    assert concurrent["fits"] == sequential["fits"]
    # Admission sampled no candidate: every candidate fit came from the
    # prefit (the seed is fitted when the novelty gate first compares with it).
    candidate_fits = [
        f for f in concurrent["fits"] if not f.startswith("bayesian_fair_coin.")
    ]
    assert concurrent["sampled_before_admission"] == candidate_fits
    # The wave exercised both orders of events it depends on.
    verdicts = {name: (ok, reason) for name, ok, reason in sequential["verdicts"]}
    assert verdicts["rep_a"][0] is True
    assert "predicts like existing model 'rep_a'" in verdicts["rep_b"][1]
    assert "forbidden import" in verdicts["rep_c"][1]
    assert "no hypothesis.md" in verdicts["rep_d"][1]
    assert "rep_a_2" in verdicts
    assert (
        "even at target_accept 0.95" in verdicts["rep_e"][1]
    )  # a near-miss refit, prefitted
