"""The design agent's CONTEXT.md must carry the current model set's hypotheses.

The design agent generates the 100-300-pair candidate pool that EIG then
scores. If the pool is generated blind to the models, EIG can only pick the
best of a blind sample — so the design CONTEXT inlines each model's hypothesis
(name + manifest rationale) and points at the implementations, letting the
agent target regions of model disagreement.
"""

from __future__ import annotations

import pytest
import yaml

from src.pipelines.outer_loop.orchestrator import write_context


def _write_models(exp_dir, models) -> None:
    models_dir = exp_dir / "cognitive_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "models_manifest.yaml").write_text(
        yaml.safe_dump({"models": models}, sort_keys=False), encoding="utf-8"
    )


def test_design_context_inlines_model_hypotheses(tmp_path):
    _write_models(
        tmp_path,
        [
            {"name": "run_length_bias", "rationale": "People count long runs."},
            {"name": "alternation_bias", "rationale": "People track alternations."},
        ],
    )
    path = write_context(tmp_path, "2_design", "subjective_randomness", 1)
    text = path.read_text(encoding="utf-8")
    assert "run_length_bias" in text
    assert "People count long runs." in text
    assert "alternation_bias" in text
    assert "People track alternations." in text


def test_design_context_missing_manifest_raises(tmp_path):
    # By design time the model set MUST exist (seeded or carried forward);
    # a missing manifest is a pipeline bug, not an empty context.
    with pytest.raises(FileNotFoundError, match="models_manifest"):
        write_context(tmp_path, "2_design", "subjective_randomness", 1)


def test_non_design_context_does_not_require_models(tmp_path):
    # Other stages (e.g. 3_implement) don't need the hypotheses section and
    # must not start failing when cognitive_models/ is absent.
    path = write_context(tmp_path, "3_implement", "subjective_randomness", 1)
    assert path.exists()
