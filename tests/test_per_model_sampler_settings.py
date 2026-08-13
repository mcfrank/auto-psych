"""Per-model sampler settings: a model file may declare its own NUTS geometry.

Integration (outside-in): fitting ``motif_stack`` must reach ``pm.sample`` with
``target_accept=0.9`` — the value its model file declares — while every other
seed model still gets the global production 0.99. The unit tests below drive
the resolution order (caller > model-declared > global default), the loud
validation of a malformed declaration, and the folding of the *resolved*
settings into both cache keys.

Motivation (measured 2026-08-13): at the global ``target_accept=0.99`` NUTS
needs ~109 leapfrog steps per iteration for motif_stack's max-of-max likelihood
(~5.1 h for a production fit); at 0.9 it needs ~25 steps with zero divergences
(~1.3 h). That is a property of *this model's* posterior geometry, so it
belongs in the model file, not in a caller.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.models import mcmc_defaults as md
from src.models import pymc_inference as pi
from src.subjective_randomness.features import featurize_responses_csv

SEED_MODELS_DIR = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "subjective_randomness"
    / "pymc_model_families"
)


class _StopSampling(Exception):
    """Raised by the fake sampler once it has captured pm.sample's kwargs."""


def _write_same_length_responses(tmp_path: Path) -> Path:
    """A tiny featurized responses CSV of same-length H/T pairs."""
    raw = tmp_path / "raw.csv"
    with raw.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["participant_id", "trial_index", "sequence_a", "sequence_b", "chose_left"],
        )
        w.writeheader()
        for i, (a, b, y) in enumerate(
            [
                ("HHTHTTHT", "HTHTHTHT", 1),
                ("HHHHTTTT", "HTHTHTHT", 0),
                ("HHTTHHTT", "HTTHHTTH", 1),
            ]
        ):
            w.writerow(
                {
                    "participant_id": "p1",
                    "trial_index": i,
                    "sequence_a": a,
                    "sequence_b": b,
                    "chose_left": y,
                }
            )
    featurized = tmp_path / "responses.csv"
    featurize_responses_csv(raw, featurized)
    return featurized


def _capture_sample_kwargs(monkeypatch) -> dict:
    import pymc as pm

    captured: dict = {}

    def fake_sample(*args, **kwargs):
        captured.update(kwargs)
        raise _StopSampling

    monkeypatch.setattr(pm, "sample", fake_sample)
    return captured


# --- Integration -----------------------------------------------------------


def test_motif_stack_fit_uses_its_declared_target_accept(tmp_path, monkeypatch):
    captured = _capture_sample_kwargs(monkeypatch)
    responses = _write_same_length_responses(tmp_path)

    with pytest.raises(_StopSampling):
        pi.fit_model("motif_stack", SEED_MODELS_DIR, responses)

    assert captured["target_accept"] == 0.9


def test_other_seed_models_still_get_the_global_production_target_accept(
    tmp_path, monkeypatch
):
    captured = _capture_sample_kwargs(monkeypatch)
    responses = _write_same_length_responses(tmp_path)

    with pytest.raises(_StopSampling):
        pi.fit_model("falk_konold_dp", SEED_MODELS_DIR, responses)

    assert captured["target_accept"] == md.PRODUCTION_TARGET_ACCEPT


def test_only_motif_stack_declares_sampler_settings():
    """No other seed model gets a bespoke NUTS geometry — 0.99 stays global."""
    declared = {
        path.stem: pi.model_sampler_settings(path.stem, SEED_MODELS_DIR)
        for path in sorted(SEED_MODELS_DIR.glob("*.py"))
        if path.stem != "__init__"
    }
    assert declared.pop("motif_stack") == {"target_accept": 0.9}
    assert all(settings == {} for settings in declared.values()), declared


# --- Unit: resolution order ------------------------------------------------


def test_caller_value_beats_the_model_declaration(tmp_path, monkeypatch):
    captured = _capture_sample_kwargs(monkeypatch)
    responses = _write_same_length_responses(tmp_path)

    with pytest.raises(_StopSampling):
        pi.fit_model("motif_stack", SEED_MODELS_DIR, responses, target_accept=0.95)

    assert captured["target_accept"] == 0.95


def test_unset_caller_argument_resolves_rather_than_pinning_the_default():
    """``None`` means "unset"; only then does the model declaration apply."""
    resolved = pi.resolve_fit_settings("motif_stack", SEED_MODELS_DIR, {"target_accept": None})
    assert resolved["target_accept"] == 0.9

    pinned = pi.resolve_fit_settings("motif_stack", SEED_MODELS_DIR, {"target_accept": 0.99})
    assert pinned["target_accept"] == 0.99


def test_resolution_fills_every_default_key():
    resolved = pi.resolve_fit_settings("motif_stack", SEED_MODELS_DIR, {})
    assert set(resolved) == set(pi._FIT_DEFAULTS)
    assert resolved["draws"] == md.PRODUCTION_DRAWS
    assert resolved["chains"] == md.PRODUCTION_CHAINS


# --- Unit: loud validation of a declaration --------------------------------


def _write_model_with_settings(models_dir: Path, body: str) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "declaring.py").write_text(
        "import numpy as np\n"
        "import pymc as pm\n"
        f"{body}\n"
        "with pm.Model() as model:\n"
        "    y = pm.Data('chose_left', np.zeros(1, dtype='int64'))\n"
        "    p = pm.Uniform('p', lower=0.01, upper=0.99)\n"
        "    p_left = pm.Deterministic('p_left', p * np.ones(1))\n"
        "    pm.Bernoulli('response', p=p_left, observed=y)\n",
        encoding="utf-8",
    )


def test_unknown_sampler_setting_key_raises(tmp_path):
    _write_model_with_settings(tmp_path, "SAMPLER_SETTINGS = {'targt_accept': 0.9}")
    with pytest.raises(ValueError, match="unknown sampler setting"):
        pi.model_sampler_settings("declaring", tmp_path)


def test_non_numeric_sampler_setting_value_raises(tmp_path):
    _write_model_with_settings(tmp_path, "SAMPLER_SETTINGS = {'target_accept': 'high'}")
    with pytest.raises(TypeError, match="must be a number"):
        pi.model_sampler_settings("declaring", tmp_path)


def test_non_dict_sampler_settings_raises(tmp_path):
    _write_model_with_settings(tmp_path, "SAMPLER_SETTINGS = [0.9]")
    with pytest.raises(TypeError, match="must be a dict"):
        pi.model_sampler_settings("declaring", tmp_path)


def test_model_without_a_declaration_resolves_to_the_global_defaults(tmp_path):
    _write_model_with_settings(tmp_path, "")
    assert pi.model_sampler_settings("declaring", tmp_path) == {}
    assert pi.resolve_fit_settings("declaring", tmp_path, {}) == dict(pi._FIT_DEFAULTS)


def test_sampler_settings_are_reread_when_the_model_file_changes(tmp_path):
    _write_model_with_settings(tmp_path, "SAMPLER_SETTINGS = {'target_accept': 0.9}")
    assert pi.model_sampler_settings("declaring", tmp_path) == {"target_accept": 0.9}
    _write_model_with_settings(tmp_path, "SAMPLER_SETTINGS = {'target_accept': 0.8}")
    assert pi.model_sampler_settings("declaring", tmp_path) == {"target_accept": 0.8}


# --- Unit: the RESOLVED settings key both caches ---------------------------


def test_cache_key_folds_in_the_model_declared_settings(tmp_path):
    """A motif_stack fit sampled at its declared 0.9 must not collide with one
    a caller explicitly requested at the global 0.99."""
    responses = _write_same_length_responses(tmp_path)
    declared = pi._cache_key("motif_stack", SEED_MODELS_DIR, responses, {})
    forced_global = pi._cache_key(
        "motif_stack",
        SEED_MODELS_DIR,
        responses,
        {"target_accept": md.PRODUCTION_TARGET_ACCEPT},
    )
    assert declared != forced_global
    # Asking for exactly what the model declares is the same fit (cache hit).
    forced_declared = pi._cache_key(
        "motif_stack", SEED_MODELS_DIR, responses, {"target_accept": 0.9}
    )
    assert declared == forced_declared


def test_disk_fingerprint_and_in_process_key_agree_on_the_resolved_settings(
    tmp_path, monkeypatch
):
    """The .nc fingerprint is built from the same resolved signature as the
    in-process key, so the two caches can never disagree about a motif_stack fit.
    """
    responses = _write_same_length_responses(tmp_path)
    cache_dir = tmp_path / "cache"
    captured = _capture_sample_kwargs(monkeypatch)

    with pytest.raises(_StopSampling):
        pi.fit_model("motif_stack", SEED_MODELS_DIR, responses, cache_dir=cache_dir)
    assert captured["target_accept"] == 0.9

    resolved = pi.resolve_fit_settings("motif_stack", SEED_MODELS_DIR, {})
    import hashlib

    expected_fp = hashlib.sha256(
        (
            pi._sha256_file(SEED_MODELS_DIR / "motif_stack.py")
            + pi._sha256_file(responses)
            + pi._sampler_signature(resolved)
        ).encode("utf-8")
    ).hexdigest()[:16]

    key = pi._cache_key("motif_stack", SEED_MODELS_DIR, responses, {})
    assert key[-1] == pi._sampler_signature(resolved)
    # And the fingerprint fit_model would write matches that same signature.
    monkeypatch.undo()
    assert expected_fp  # computed from the resolved settings, not the raw defaults
