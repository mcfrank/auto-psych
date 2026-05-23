"""Task 3: persistent model zoo."""

import json

import numpy as np
import pytest

from src.pipelines.inner_loop.fitting import FitResult
from src.pipelines.inner_loop.zoo import (
    INITIAL_ENTRY_ID,
    candidate_entry_id,
    get_entry,
    iter_zoo,
    record_candidate,
    record_entry,
    record_initial,
    zoo_size,
)


def _fit(ll: float = -1.0, n_params: int = 1) -> FitResult:
    return FitResult(
        params=[0.5] * n_params if n_params else [],
        log_likelihood=ll,
        per_trial_ll=np.array([ll]),
        n_samples=1,
        n_trials=1,
        n_params=n_params,
    )


def _write_dummy_model(path):
    path.write_text("def cognitive_model(s, t, p): return None\n")


# ---- entry id naming ---------------------------------------------------------


def test_candidate_entry_id_is_deterministic():
    assert candidate_entry_id(0, "candidate_0") == "iter_0__candidate_0"
    assert candidate_entry_id(7, "candidate_3") == "iter_7__candidate_3"


# ---- record_entry basics ------------------------------------------------------


def test_record_entry_writes_model_fit_and_origin(tmp_path):
    src_model = tmp_path / "src_model.py"
    _write_dummy_model(src_model)
    fit = _fit(ll=-2.5, n_params=2)

    entry = record_entry(
        tmp_path / "zoo",
        "test_entry",
        src_model,
        fit,
        origin={"note": "hello"},
    )

    assert entry.dir.is_dir()
    assert entry.model_path.read_text() == src_model.read_text()
    payload = json.loads(entry.fit_path.read_text())
    assert payload["log_likelihood"] == -2.5
    assert payload["n_params"] == 2

    origin = json.loads(entry.origin_path.read_text())
    assert origin["entry_id"] == "test_entry"
    assert origin["note"] == "hello"
    assert "recorded_at" in origin


def test_record_entry_is_idempotent_for_same_id(tmp_path):
    """Resume safety: re-recording the same id overwrites cleanly."""
    src = tmp_path / "m.py"
    _write_dummy_model(src)

    record_entry(tmp_path / "zoo", "x", src, _fit(ll=-1.0))
    record_entry(tmp_path / "zoo", "x", src, _fit(ll=-2.0))

    entry = get_entry(tmp_path / "zoo", "x")
    assert entry is not None
    payload = json.loads(entry.fit_path.read_text())
    assert payload["log_likelihood"] == -2.0
    assert zoo_size(tmp_path / "zoo") == 1


def test_record_entry_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        record_entry(tmp_path / "zoo", "x", tmp_path / "nope.py", _fit())


# ---- candidate / initial wrappers --------------------------------------------


def test_record_candidate_uses_canonical_id(tmp_path):
    src = tmp_path / "m.py"
    _write_dummy_model(src)
    entry = record_candidate(tmp_path / "zoo", 2, "candidate_1", src, _fit())
    assert entry.entry_id == "iter_2__candidate_1"

    origin = json.loads(entry.origin_path.read_text())
    assert origin["iteration"] == 2
    assert origin["candidate_id"] == "candidate_1"


def test_record_initial_uses_initial_entry_id(tmp_path):
    src = tmp_path / "m.py"
    _write_dummy_model(src)
    entry = record_initial(tmp_path / "zoo", src, _fit())
    assert entry.entry_id == INITIAL_ENTRY_ID

    origin = json.loads(entry.origin_path.read_text())
    assert origin["role"] == "initial_model"


# ---- iteration ---------------------------------------------------------------


def test_iter_zoo_returns_sorted_well_formed_entries(tmp_path):
    src = tmp_path / "m.py"
    _write_dummy_model(src)
    zoo = tmp_path / "zoo"

    record_candidate(zoo, 1, "candidate_0", src, _fit())
    record_candidate(zoo, 0, "candidate_1", src, _fit())
    record_initial(zoo, src, _fit())

    ids = [e.entry_id for e in iter_zoo(zoo)]
    assert ids == sorted(ids)
    assert set(ids) == {
        "iter_0__candidate_1",
        "iter_1__candidate_0",
        INITIAL_ENTRY_ID,
    }


def test_iter_zoo_skips_partial_entries(tmp_path):
    """A directory missing fit_result.json is silently skipped, not an error."""
    src = tmp_path / "m.py"
    _write_dummy_model(src)
    zoo = tmp_path / "zoo"
    record_candidate(zoo, 0, "candidate_0", src, _fit())

    partial = zoo / "iter_99__candidate_0"
    partial.mkdir(parents=True)
    (partial / "cognitive_model.py").write_text("x = 1")
    # no fit_result.json

    assert zoo_size(zoo) == 1
    assert {e.entry_id for e in iter_zoo(zoo)} == {"iter_0__candidate_0"}


def test_iter_zoo_empty_or_missing_zoo(tmp_path):
    assert list(iter_zoo(tmp_path / "does_not_exist")) == []
    (tmp_path / "empty_zoo").mkdir()
    assert list(iter_zoo(tmp_path / "empty_zoo")) == []


# ---- load_fit roundtrip ------------------------------------------------------


def test_zoo_entry_load_fit_returns_fit_result(tmp_path):
    src = tmp_path / "m.py"
    _write_dummy_model(src)
    fit = _fit(ll=-7.0, n_params=3)
    record_candidate(tmp_path / "zoo", 0, "candidate_0", src, fit)

    entry = next(iter_zoo(tmp_path / "zoo"))
    loaded = entry.load_fit()
    assert isinstance(loaded, FitResult)
    assert loaded.log_likelihood == -7.0
    assert loaded.n_params == 3
    np.testing.assert_array_equal(loaded.per_trial_ll, fit.per_trial_ll)
