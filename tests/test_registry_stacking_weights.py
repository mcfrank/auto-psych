"""The registry (the EIG model prior for the next experiment's design) must
record arviz's stacking weights, not the softmax posterior.

The softmax of total ELPD-LOO is knowingly overconfident (~1.0 for one model
even when rivals are within noise; see model_comparison/posterior.py), and an
overconfident design prior makes EIG optimize discrimination around a single
model. az.compare's stacking weights are computed exactly for weighting
predictive distributions and are already persisted in model_posterior.json's
``comparison`` block — the registry must read those. A missing or malformed
posterior export fails loudly: the registry is only updated after a model loop
ran, so its absence means the pipeline is broken, not "nothing to record".
"""

from __future__ import annotations

import json

import pytest
import yaml

from src.pipelines.outer_loop.orchestrator import update_registry_from_interpretation


def _write_posterior(exp_dir, payload) -> None:
    loop_dir = exp_dir / "model_loop"
    loop_dir.mkdir(parents=True, exist_ok=True)
    (loop_dir / "model_posterior.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _read_theories(exp_dir) -> dict:
    registry = yaml.safe_load(
        (exp_dir / "model_registry.yaml").read_text(encoding="utf-8")
    )
    return registry["theories"]


def _payload(weights, posteriors=None):
    names = list(weights)
    return {
        # Deliberately different from the stacking weights: the overconfident
        # softmax the registry must NOT copy.
        "posteriors": posteriors or {names[0]: 0.98, names[1]: 0.02},
        "elpd_loo": {name: -10.0 - i for i, name in enumerate(names)},
        "comparison": {
            name: {
                "rank": i,
                "elpd_loo": -10.0 - i,
                "elpd_diff": float(i),
                "dse": 2.0,
                "weight": w,
                "loo_unreliable": False,
            }
            for i, (name, w) in enumerate(weights.items())
        },
    }


def test_registry_records_stacking_weights_not_softmax(tmp_path):
    _write_posterior(tmp_path, _payload({"a": 0.6, "b": 0.4}))
    update_registry_from_interpretation(tmp_path)
    theories = _read_theories(tmp_path)
    assert theories["a"] == pytest.approx(0.6)
    assert theories["b"] == pytest.approx(0.4)


def test_registry_normalizes_stacking_weights(tmp_path):
    # az.compare weights sum to 1 already, but the registry must not rely on it.
    _write_posterior(tmp_path, _payload({"a": 0.3, "b": 0.1}))
    update_registry_from_interpretation(tmp_path)
    theories = _read_theories(tmp_path)
    assert theories["a"] == pytest.approx(0.75)
    assert theories["b"] == pytest.approx(0.25)


def test_registry_missing_posterior_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="model_posterior"):
        update_registry_from_interpretation(tmp_path)


def test_registry_malformed_posterior_json_raises(tmp_path):
    loop_dir = tmp_path / "model_loop"
    loop_dir.mkdir(parents=True)
    (loop_dir / "model_posterior.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="model_posterior"):
        update_registry_from_interpretation(tmp_path)


def test_registry_missing_comparison_block_raises(tmp_path):
    _write_posterior(
        tmp_path, {"posteriors": {"a": 0.9, "b": 0.1}, "elpd_loo": {"a": -1, "b": -2}}
    )
    with pytest.raises(ValueError, match="comparison"):
        update_registry_from_interpretation(tmp_path)


def test_registry_all_zero_weights_raise(tmp_path):
    _write_posterior(tmp_path, _payload({"a": 0.0, "b": 0.0}))
    with pytest.raises(ValueError, match="weight"):
        update_registry_from_interpretation(tmp_path)
