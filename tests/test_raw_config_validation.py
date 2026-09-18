"""Pool models that cannot bind a raw stimulus row must fail at config
resolution, naming the model and the missing columns.

Every seed model defines ``compute_features`` or ``prepare_observed`` hooks.
A model that expects precomputed columns from the CSV fails loudly before the
run starts.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.pipelines.outer_loop.columns import RAW_RESPONSE_COLUMNS


# ─── helpers ───

_RAW_ROW = {c: "0" for c in RAW_RESPONSE_COLUMNS}

# A model that reads a precomputed column the raw CSV does not carry.
_FEATURIZED_MODEL_CODE = """\
import numpy as np
import pymc as pm

with pm.Model() as model:
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(n_a.astype("float64")))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
"""

# A model with compute_features that supplies its own columns.
_RAW_MODEL_CODE = """\
import numpy as np
import pymc as pm

def compute_features(sequence_a, sequence_b):
    return {"my_feat": float(len(sequence_a))}

with pm.Model() as model:
    my_feat = pm.Data("my_feat", np.zeros(1, dtype="float64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(my_feat))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
"""

# A model with prepare_observed that is self-contained.
_RAW_PREP_MODEL_CODE = """\
import numpy as np
import pymc as pm

def prepare_observed(rows):
    return {"my_feat": np.array([len(r["sequence_a"]) for r in rows], dtype="float64"),
            "chose_left": np.array([int(r["chose_left"]) for r in rows], dtype="int64")}

with pm.Model() as model:
    my_feat = pm.Data("my_feat", np.zeros(1, dtype="float64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(my_feat))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
"""


def _write_model_dir(path: Path, models: dict[str, str]):
    path.mkdir(parents=True, exist_ok=True)
    entries = [{"name": name, "rationale": "test"} for name in models]
    (path / "models_manifest.yaml").write_text(
        yaml.safe_dump({"models": entries}), encoding="utf-8"
    )
    for name, code in models.items():
        (path / f"{name}.py").write_text(code, encoding="utf-8")


# ─── tests ───


def test_featurized_model_in_raw_pool_raises_naming_model_and_columns(tmp_path):
    """A raw config with a pool model that cannot bind a raw row raises at
    config resolution, naming the model and the missing columns."""
    from src.subjective_randomness.holdout_data import (
        validate_raw_pool_models,
    )

    pool = tmp_path / "pool"
    _write_model_dir(pool, {"bad_model": _FEATURIZED_MODEL_CODE})

    with pytest.raises(ValueError, match="bad_model") as exc_info:
        validate_raw_pool_models(pool)
    assert "n_a" in str(exc_info.value)


def test_raw_model_with_compute_features_passes(tmp_path):
    from src.subjective_randomness.holdout_data import (
        validate_raw_pool_models,
    )

    pool = tmp_path / "pool"
    _write_model_dir(pool, {"good_model": _RAW_MODEL_CODE})
    validate_raw_pool_models(pool)


def test_raw_model_with_prepare_observed_passes(tmp_path):
    from src.subjective_randomness.holdout_data import (
        validate_raw_pool_models,
    )

    pool = tmp_path / "pool"
    _write_model_dir(pool, {"good_model": _RAW_PREP_MODEL_CODE})
    validate_raw_pool_models(pool)


def test_actual_seed_dirs_pass():
    """The shipped seed directories must pass the pool validation."""
    from src.subjective_randomness.holdout_data import (
        validate_raw_pool_models,
    )
    from tests.paths import REPO_ROOT

    registry = (
        REPO_ROOT / "src" / "subjective_randomness" / "pymc_model_families"
    )
    pool = (
        REPO_ROOT
        / "src"
        / "pipelines"
        / "outer_loop"
        / "projects"
        / "subjective_randomness"
        / "seed_models"
    )

    validate_raw_pool_models(registry)
    validate_raw_pool_models(pool)


def test_pool_validation_catches_featurized_model(tmp_path):
    """A pool model that expects precomputed columns is caught by validation."""
    from src.subjective_randomness.holdout_data import (
        validate_raw_pool_models,
    )

    pool = tmp_path / "pool"
    _write_model_dir(
        pool, {"featurized_model": _FEATURIZED_MODEL_CODE}
    )

    with pytest.raises(ValueError, match="featurized_model"):
        validate_raw_pool_models(pool)
