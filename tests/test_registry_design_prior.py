"""The registry (the EIG model prior for the next experiment's design) is a
uniform prior over the carried model set.

It used to copy ``az.compare``'s stacking weights. Those are ensemble
coefficients, not plausibility — a model 1.6 nats behind the best read 0.000
because its predictions were redundant with the best's, one 95 nats behind
read 0.33 because they differed — and they were written over the whole zoo,
including models never carried into the next experiment. In 15 of the 40
next-experiment designs of the iteration-2 recovery sweep every model actually
present had weight ~0, or one had weight 1.0: a degenerate prior under which
all 32 EIG-selected stimuli had zero EIG. The carried set is the protected
seeds plus every model still within the pruning margin of the best, so a
uniform prior over it asks the design to separate exactly the unresolved
hypotheses. A missing posterior export or an absent/empty carried set fails
loudly: the registry is only updated after a model loop ran and exported.
"""

from __future__ import annotations

import json

import pytest
import yaml

from src.pipelines.outer_loop.orchestrator import update_registry_from_interpretation


def _write_posterior(exp_dir, weights) -> None:
    loop_dir = exp_dir / "model_loop"
    loop_dir.mkdir(parents=True, exist_ok=True)
    names = list(weights)
    (loop_dir / "model_posterior.json").write_text(
        json.dumps(
            {
                "best_model": names[0],
                "posteriors": {n: (1.0 if n == names[0] else 0.0) for n in names},
                "elpd_loo": {n: -10.0 - i for i, n in enumerate(names)},
                "comparison": {
                    n: {"rank": i, "elpd_loo": -10.0 - i, "elpd_diff": float(i),
                        "dse": 2.0, "weight": w, "loo_unreliable": False}
                    for i, (n, w) in enumerate(weights.items())
                },
            }
        ),
        encoding="utf-8",
    )


def _write_carried_set(exp_dir, names) -> None:
    cog_dir = exp_dir / "cognitive_models"
    cog_dir.mkdir(parents=True, exist_ok=True)
    (cog_dir / "models_manifest.yaml").write_text(
        yaml.safe_dump(
            {"models": [{"name": n, "rationale": f"mechanism {n}"} for n in names]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _read_registry(exp_dir) -> dict:
    return yaml.safe_load((exp_dir / "model_registry.yaml").read_text(encoding="utf-8"))


def test_registry_is_uniform_over_the_carried_set_not_the_stacking_weights(tmp_path):
    # Stacking put everything on one model (the degenerate case) and the zoo
    # held a model that was not carried; neither reaches the registry.
    _write_posterior(tmp_path, {"winner": 1.0, "rival": 0.0, "uncarried": 0.0})
    _write_carried_set(tmp_path, ["seed_a", "seed_b", "winner", "rival"])

    update_registry_from_interpretation(tmp_path)

    registry = _read_registry(tmp_path)
    assert registry["theories"] == pytest.approx(
        {"seed_a": 0.25, "seed_b": 0.25, "winner": 0.25, "rival": 0.25}
    )
    assert registry["reserved_for_new"] == 0.0


def test_registry_missing_posterior_file_raises(tmp_path):
    _write_carried_set(tmp_path, ["seed_a"])
    with pytest.raises(FileNotFoundError, match="model_posterior"):
        update_registry_from_interpretation(tmp_path)


def test_registry_missing_carried_set_raises(tmp_path):
    _write_posterior(tmp_path, {"winner": 1.0})
    with pytest.raises(FileNotFoundError, match="models_manifest"):
        update_registry_from_interpretation(tmp_path)


def test_registry_empty_carried_set_raises(tmp_path):
    _write_posterior(tmp_path, {"winner": 1.0})
    _write_carried_set(tmp_path, [])
    with pytest.raises(ValueError, match="lists no models"):
        update_registry_from_interpretation(tmp_path)
