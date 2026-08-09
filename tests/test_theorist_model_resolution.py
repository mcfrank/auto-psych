"""Fail-loud policy for resolving and running theorist models.

``get_model_predictions`` used to swallow every exception per model and simply
omit it from the returned dict. Downstream that is invisible and actively
wrong: EIG renormalizes over whichever models happened to load, so a model with
a typo quietly stops counting as a hypothesis instead of announcing itself.

``get_model_callable``'s "only public callable" fallback stays (a model file
whose entry point is named something else is recoverable) but must say which
symbol it bound, since binding the wrong one silently changes every prediction.
"""

from __future__ import annotations

import logging

import pytest

from src.models.theorist.loader import get_model_callable, get_model_names_from_manifest
from src.models.theorist.predictions import get_model_predictions

STIMULUS = ("HHTT", "HTHT")
RESPONSES = ["left", "right"]


def _write_model(theorist_dir, name: str, body: str):
    theorist_dir.mkdir(parents=True, exist_ok=True)
    (theorist_dir / f"{name}.py").write_text(body, encoding="utf-8")
    return theorist_dir


def test_predictions_raise_when_a_model_cannot_be_loaded(tmp_path):
    with pytest.raises(FileNotFoundError, match="absent_model"):
        get_model_predictions(STIMULUS, RESPONSES, ["absent_model"], tmp_path)


def test_manifest_resolution_raises_when_a_listed_model_file_is_missing(tmp_path):
    manifest = {"models": [{"name": "absent_model"}]}
    with pytest.raises(FileNotFoundError, match="absent_model"):
        get_model_names_from_manifest(manifest, tmp_path)


def test_predictions_raise_when_no_theorist_dir_is_given():
    with pytest.raises(KeyError, match="theorist_dir required"):
        get_model_predictions(STIMULUS, RESPONSES, ["any_model"], None)


def test_predictions_raise_when_a_model_crashes_while_predicting(tmp_path):
    _write_model(
        tmp_path,
        "explodes",
        "def explodes(stimulus, response_options):\n"
        "    raise ZeroDivisionError('bad math in the model')\n",
    )
    with pytest.raises(ZeroDivisionError, match="bad math"):
        get_model_predictions(STIMULUS, RESPONSES, ["explodes"], tmp_path)


def test_predictions_still_return_every_working_model(tmp_path):
    for name in ("m_one", "m_two"):
        _write_model(
            tmp_path,
            name,
            f"def {name}(stimulus, response_options):\n"
            "    return {'left': 0.5, 'right': 0.5}\n",
        )
    out = get_model_predictions(STIMULUS, RESPONSES, ["m_one", "m_two"], tmp_path)
    assert sorted(out) == ["m_one", "m_two"]


def test_sole_public_callable_fallback_is_logged(tmp_path, caplog):
    _write_model(
        tmp_path,
        "oddly_named",
        "def predict(stimulus, response_options):\n"
        "    return {'left': 0.5, 'right': 0.5}\n",
    )
    with caplog.at_level(logging.WARNING, logger="src.models.theorist.loader"):
        fn = get_model_callable("oddly_named", tmp_path)
    assert fn(STIMULUS, RESPONSES) == {"left": 0.5, "right": 0.5}
    assert "predict" in caplog.text and "oddly_named" in caplog.text
