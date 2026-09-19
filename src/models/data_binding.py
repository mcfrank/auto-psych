"""Data binding for PyMC cognitive models.

Reads response CSVs, maps columns to ``pm.Data`` containers, runs optional
model hooks (``compute_features``, ``prepare_observed``), and produces the
``{name: np.ndarray}`` dicts that ``pm.set_data`` consumes.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.models.model_loading import (
    _COMPUTE_FEATURES_ATTR,
    _PREPARE_OBSERVED_ATTR,
    observed_response_data,
    pm_data_inputs,
)


# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------

def _read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    """Read a CSV file into a list of row dicts."""
    with Path(csv_path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Feature hooks
# ---------------------------------------------------------------------------

def _model_compute_features(model: Any) -> Any:
    """The model's optional ``compute_features`` callable, or ``None``."""
    return getattr(model, _COMPUTE_FEATURES_ATTR, None)


# Columns a model's compute_features may never return: the observed response
# and the row bookkeeping. A model that redefines one of these is wrong
# regardless of the value it produces.
PROTECTED_ROW_COLUMNS = frozenset(
    {"chose_left", "participant_id", "trial_index", "sequence_a", "sequence_b"}
)


def _same_feature_value(existing: Any, computed: Any) -> bool:
    """Whether a model's computed feature value agrees with a column already present.

    When the evaluation harness pre-populates feature columns (via
    ``feature_rows``), a model's ``compute_features`` hook may recompute an
    already-present column. Producing the same value is harmless; producing a
    DIFFERENT value under the same name means the model silently redefines
    a harness column, which must fail loudly.
    """
    try:
        return math.isclose(float(existing), float(computed), rel_tol=1e-9, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def _augment_rows_with_features(
    model: Any, rows: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Add a model's ``compute_features`` columns to each row.

    Runs the model's ``compute_features(sequence_a, sequence_b)`` hook over
    every row's raw H/T sequences and merges the returned columns. A no-op
    (returns ``rows`` unchanged) for models that do not declare the hook.
    Fails loudly if the hook is declared but rows lack the raw sequences,
    returns non-dict/non-numeric values, varies its keys across rows, or
    collides with a protected column.
    """
    compute_fn = _model_compute_features(model)
    if compute_fn is None or not rows:
        return rows

    missing = {"sequence_a", "sequence_b"} - set(rows[0].keys())
    if missing:
        raise ValueError(
            f"Model declares compute_features but rows are missing "
            f"{sorted(missing)}; the raw H/T sequence columns are required to "
            "compute extra features."
        )

    augmented: List[Dict[str, Any]] = []
    expected_keys: Optional[tuple] = None
    for i, r in enumerate(rows):
        extra = compute_fn(r["sequence_a"], r["sequence_b"])
        if not isinstance(extra, dict):
            raise TypeError(
                f"compute_features must return a dict of feature_name -> number, "
                f"got {type(extra).__name__} for row {i}."
            )
        keys = tuple(sorted(extra.keys()))
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise ValueError(
                "compute_features returned inconsistent feature names: row 0 -> "
                f"{list(expected_keys)}, row {i} -> {list(keys)}. It must return "
                "the same feature names for every stimulus."
            )
        for name, value in extra.items():
            if name in PROTECTED_ROW_COLUMNS:
                raise ValueError(
                    f"compute_features returned {name!r}, which collides with the "
                    "response/bookkeeping columns a row must keep "
                    f"({sorted(PROTECTED_ROW_COLUMNS)}); extra features must use "
                    "new names."
                )
            if name in r and not _same_feature_value(r[name], value):
                raise ValueError(
                    f"compute_features feature {name!r} collides with an existing "
                    f"column that holds a DIFFERENT value ({r[name]!r} vs "
                    f"{value!r}); a model may recompute a column the harness also "
                    "supplies, but it may not redefine what the name means."
                )
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"compute_features feature {name!r} must be a number, got "
                    f"{type(value).__name__} ({value!r})."
                )
            if not math.isfinite(float(value)):
                raise ValueError(
                    f"compute_features feature {name!r} is not finite ({value!r})."
                )
        augmented.append({**r, **extra})
    return augmented


# ---------------------------------------------------------------------------
# prepare_observed hook
# ---------------------------------------------------------------------------

def _model_prepare_observed(model: Any) -> Any:
    """The model's optional ``prepare_observed`` callable, or ``None``."""
    return getattr(model, _PREPARE_OBSERVED_ATTR, None)


def _observed_via_hook(model: Any, rows: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Build every ``pm.Data`` array through the model's ``prepare_observed`` hook.

    The hook owns the layout, so the harness cannot check it column by column.
    It checks the contract instead, and fails loudly on any breach — a hook that
    returned the wrong keys or a mis-shaped array would otherwise surface much
    later as an inscrutable pytensor shape error, or worse, as a silently
    mis-aligned likelihood:

    - the returned keys must be exactly the model's ``pm.Data`` names;
    - every value must be a numpy array whose rank matches the container's
      placeholder, and whose dtype has the same *kind* (a float array for an
      integer container would be truncated on binding);
    - the observed-response container must have one entry per input row, which
      is what makes ``p_left`` per-trial and keeps ELPD-LOO pointwise.

    Exact dtype *width* is normalized here rather than demanded of the hook,
    because the width is PyMC's choice, not the model's: ``pm.Data`` converts an
    ``int64`` array to ``intX`` (int32 under PyMC 5.28), so a hook that hard-coded
    a width would break on a different build. This mirrors what the
    column-mapping path does with ``placeholder.dtype``.
    """
    prepare = _model_prepare_observed(model)
    if prepare is None:
        raise ValueError("Model declares no prepare_observed hook.")
    out = prepare(list(rows))
    if not isinstance(out, dict):
        raise TypeError(
            "prepare_observed must return a dict of pm.Data name -> numpy array, "
            f"got {type(out).__name__}."
        )
    expected = set(pm_data_inputs(model))
    got = set(out)
    if got != expected:
        missing = sorted(expected - got)
        unexpected = sorted(got - expected)
        raise ValueError(
            "prepare_observed must return exactly the model's pm.Data containers. "
            f"Missing: {missing}. Unexpected: {unexpected}."
        )
    bound: Dict[str, np.ndarray] = {}
    for name in sorted(out):
        arr = out[name]
        if not isinstance(arr, np.ndarray):
            raise TypeError(
                f"prepare_observed returned {type(arr).__name__} for {name!r}; "
                "every value must be a numpy array."
            )
        placeholder = model.named_vars[name].get_value()
        if arr.ndim != placeholder.ndim:
            raise ValueError(
                f"prepare_observed returned a {arr.ndim}-D array for {name!r} but "
                f"its pm.Data placeholder is {placeholder.ndim}-D."
            )
        if arr.dtype.kind != placeholder.dtype.kind:
            raise ValueError(
                f"prepare_observed returned dtype {arr.dtype} for {name!r} but its "
                f"pm.Data placeholder is {placeholder.dtype} — the kinds differ, so "
                "binding would silently reinterpret the values."
            )
        cast = arr.astype(placeholder.dtype, copy=False)
        if not np.array_equal(cast, arr):
            raise ValueError(
                f"prepare_observed's {name!r} array does not survive the cast to the "
                f"container's dtype {placeholder.dtype} (values out of range)."
            )
        bound[name] = cast
    out = bound
    response_name = observed_response_data(model)
    n_response = len(out[response_name])
    if n_response != len(rows):
        raise ValueError(
            f"prepare_observed returned {n_response} entries for the observed-response "
            f"container {response_name!r} but was given {len(rows)} rows; the observed "
            "response must stay one-per-trial."
        )
    return out


# ---------------------------------------------------------------------------
# Non-stimulus columns & MissingStimulusColumns
# ---------------------------------------------------------------------------

# Columns a *response* row carries but a bare stimulus row never does. A model
# that binds only these beyond the features is legitimately unevaluable on a
# stimulus (a participant-level random effect, say) and may be screened out of
# a design; anything else missing means the rows were built wrong.
NON_STIMULUS_COLUMNS = frozenset({"participant_id", "trial_index"})


class MissingStimulusColumns(ValueError):
    """A model needs columns the given rows do not carry.

    Subclasses ``ValueError`` so existing handlers still catch it, but exposes
    ``missing`` and ``available`` as data. Callers that must decide *why* a
    model would not bind — the EIG screen distinguishes a legitimate
    participant-level mismatch from rows built without a featurizer — need that
    structurally, not by re-parsing a formatted message.
    """

    def __init__(self, missing: Sequence[str], available: Sequence[str]):
        self.missing = tuple(missing)
        self.available = tuple(available)
        super().__init__(
            f"Rows missing columns {list(self.missing)} required by the model. "
            f"Available: {list(self.available)}"
        )

    @property
    def only_non_stimulus(self) -> bool:
        """True when every missing column is response-row bookkeeping."""
        return bool(self.missing) and set(self.missing) <= NON_STIMULUS_COLUMNS


# ---------------------------------------------------------------------------
# Public data-binding API
# ---------------------------------------------------------------------------

def make_stim_data(model: Any, rows: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Build a `pm.set_data` dict from a list of row dicts for a given model.

    Each `pm.Data` container in `model` is filled with the corresponding column
    from `rows`, cast to the placeholder's dtype. Useful for predict_p_left and
    sample_synthetic_responses, where the caller has rows but not a CSV file.

    If the model declares a ``prepare_observed`` hook it builds every container
    itself and the column mapping is skipped. Otherwise, if the model declares a
    ``compute_features`` hook, its extra columns are computed from each row's
    raw sequences first.
    """
    if _model_prepare_observed(model) is not None:
        return _observed_via_hook(model, rows)
    rows = _augment_rows_with_features(model, rows)
    inputs = pm_data_inputs(model)
    missing = [c for c in inputs if rows and c not in rows[0]]
    if missing:
        raise MissingStimulusColumns(missing, list(rows[0].keys()) if rows else [])
    out: Dict[str, np.ndarray] = {}
    for col in inputs:
        placeholder = model.named_vars[col].get_value()
        dtype = placeholder.dtype
        values = [r[col] for r in rows]
        if np.issubdtype(dtype, np.integer):
            arr = np.array([int(float(v)) for v in values], dtype=dtype)
        elif np.issubdtype(dtype, np.floating):
            arr = np.array([float(v) for v in values], dtype=dtype)
        else:
            arr = np.array(values, dtype=dtype)
        out[col] = arr
    return out


def extract_observed(csv_path: Path, model: Any) -> Dict[str, np.ndarray]:
    """Read csv_path and pull one numpy array per pm.Data container in the model.

    Dtype is inferred from the model's current pm.Data placeholder (int64,
    float64, etc.). Fails loudly if any expected column is missing.

    A model that declares a ``prepare_observed`` hook builds its containers from
    the raw CSV rows instead (see :func:`_observed_via_hook`).
    """
    rows = _read_csv_rows(csv_path)
    if not rows:
        raise ValueError(f"No rows in {csv_path}")
    if _model_prepare_observed(model) is not None:
        return _observed_via_hook(model, rows)
    rows = _augment_rows_with_features(model, rows)

    inputs = pm_data_inputs(model)
    missing = [c for c in inputs if c not in rows[0]]
    if missing:
        raise ValueError(
            f"Responses CSV {csv_path} is missing columns {missing} required by the model. "
            f"Available columns: {list(rows[0].keys())}"
        )

    out: Dict[str, np.ndarray] = {}
    for col in inputs:
        placeholder = model.named_vars[col].get_value()
        dtype = placeholder.dtype
        values = [r[col] for r in rows]
        if np.issubdtype(dtype, np.integer):
            arr = np.array([int(float(v)) for v in values], dtype=dtype)
        elif np.issubdtype(dtype, np.floating):
            arr = np.array([float(v) for v in values], dtype=dtype)
        else:
            arr = np.array(values, dtype=dtype)
        out[col] = arr
    return out
