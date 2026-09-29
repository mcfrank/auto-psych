"""The model contract (second audit B13; first audit D5).

Every model is scored on its observed variable (ELPD-LOO) but used through its
``p_left`` (design, novelty gate, evaluation). Nothing used to check that the
two are the same thing, or that the observed variable is the data at all.
``model_contract_violation`` checks, without sampling, that:

- the observed data bound from a response file are its ``chose_left`` column,
  in row order;
- ``p_left`` has one entry per trial;
- the likelihood's per-trial probability of the observed response is
  Bernoulli(chose_left; p_left), at the initial point and at jittered points.

It runs at admission (``_cheap_gate_rejection``, so the wave prefit too), at
the starting-model screen, and in the candidate self-check. Every project
starting model and every ground truth passes it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.models.model_manifest import read_manifest_entries
from src.models.model_contract import (
    CONTRACT_PROBABILITY_TOLERANCE,
    model_contract_violation,
)
from src.pipelines.inner_loop import model_zoo
from src.pipelines.inner_loop.check_candidate import (
    CandidateCheckFailed,
    run_candidate_check,
)
from src.pipelines.inner_loop.hypothesis_ledger import HypothesisLedger
from tests.paths import REPO_ROOT

_GOOD = '''
import numpy as np
import pymc as pm


def compute_features(sequence_a, sequence_b):
    return {"h_diff": (sequence_a.count("H") - sequence_b.count("H")) / len(sequence_a)}


with pm.Model() as model:
    h_diff = pm.Data("h_diff", np.zeros(1))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    beta = pm.Normal("beta", 0.0, 1.0)
    bias = pm.Normal("bias", 0.0, 1.0)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(beta * h_diff + bias))
    pm.Bernoulli("obs", p=p_left, observed=chose_left)
'''

# Fitted to the opposite of the responses.
_FLIPPED = _GOOD.replace("observed=chose_left", "observed=1 - chose_left")

# Reorders the rows it binds: the ELPD total is unchanged, the pointwise LOO
# values no longer line up with the CSV rows.
_REORDERED = '''
import numpy as np
import pymc as pm


def prepare_observed(rows):
    rows = list(rows)[::-1]
    return {
        "h_diff": np.array(
            [(r["sequence_a"].count("H") - r["sequence_b"].count("H")) / len(r["sequence_a"])
             for r in rows]
        ),
        "chose_left": np.array(
            [int(float(r.get("chose_left", 0))) for r in rows], dtype="int64"
        ),
    }


with pm.Model() as model:
    h_diff = pm.Data("h_diff", np.zeros(1))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    beta = pm.Normal("beta", 0.0, 1.0)
    bias = pm.Normal("bias", 0.0, 1.0)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(beta * h_diff + bias))
    pm.Bernoulli("obs", p=p_left, observed=chose_left)
'''

# A lapse applied after p_left: scored on one probability, used through another.
_LAPSE_AFTER_P_LEFT = _GOOD.replace(
    'pm.Bernoulli("obs", p=p_left, observed=chose_left)',
    'pm.Bernoulli("obs", p=0.9 * p_left + 0.05, observed=chose_left)',
)

# p_left is one number, not one per trial.
_SCALAR_P_LEFT = _GOOD.replace(
    'p_left = pm.Deterministic("p_left", pm.math.sigmoid(beta * h_diff + bias))',
    'p_left = pm.Deterministic("p_left", pm.math.sigmoid(bias))',
)

_NO_P_LEFT = _GOOD.replace('pm.Deterministic("p_left", ', 'pm.Deterministic("p_choice", ')

# A likelihood term outside the observed variable: LOO never sees it.
_RESPONSE_POTENTIAL = _GOOD.replace(
    'pm.Bernoulli("obs", p=p_left, observed=chose_left)',
    'pm.Bernoulli("obs", p=p_left, observed=chose_left)\n'
    '    pm.Potential("extra", pm.math.sum(chose_left * beta))',
)

# Equivalent parameterisations that must pass: the logit form, a numerical
# clip guard far inside the tolerance, and a parameter-only Potential.
_LOGIT_FORM = _GOOD.replace(
    'pm.Bernoulli("obs", p=p_left, observed=chose_left)',
    'pm.Bernoulli("obs", logit_p=beta * h_diff + bias, observed=chose_left)',
)
_CLIP_GUARD = _GOOD.replace(
    'pm.Bernoulli("obs", p=p_left, observed=chose_left)',
    'pm.Bernoulli("obs", p=pm.math.clip(p_left, 1e-9, 1 - 1e-9), observed=chose_left)',
)
_PARAMETER_POTENTIAL = _GOOD.replace(
    'pm.Bernoulli("obs", p=p_left, observed=chose_left)',
    'pm.Bernoulli("obs", p=p_left, observed=chose_left)\n'
    '    pm.Potential("shrink", -0.5 * beta ** 2)',
)

_ROWS = [
    ("HHTTHT", "HTHTHT", 1, 0, 1),
    ("HHHHTT", "HTTHTH", 1, 1, 0),
    ("HTHTHT", "HHHHHH", 2, 0, 1),
    ("TTTHHT", "HTHHTH", 2, 1, 1),
    ("HHHTTT", "THTHHT", 3, 0, 0),
]


def _write_responses(tmp_path: Path) -> Path:
    path = tmp_path / "responses.csv"
    lines = ["sequence_a,sequence_b,participant_id,trial_index,chose_left"]
    lines += [",".join(str(v) for v in row) for row in _ROWS]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_model(models_dir: Path, name: str, source: str) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / f"{name}.py").write_text(source, encoding="utf-8")


def _violation(tmp_path: Path, source: str) -> str | None:
    _write_model(tmp_path / "models", "m", source)
    return model_contract_violation("m", tmp_path / "models", _write_responses(tmp_path))


@pytest.mark.parametrize(
    "source", [_GOOD, _LOGIT_FORM, _CLIP_GUARD, _PARAMETER_POTENTIAL],
    ids=["plain", "logit_p", "clip_guard", "parameter_potential"],
)
def test_a_model_honouring_the_contract_passes(tmp_path, source):
    assert _violation(tmp_path, source) is None


def test_observed_data_that_are_not_chose_left_fail(tmp_path):
    reason = _violation(tmp_path, _FLIPPED)
    assert reason is not None
    assert "chose_left" in reason and "observed" in reason


def test_observed_data_in_another_row_order_fail(tmp_path):
    reason = _violation(tmp_path, _REORDERED)
    assert reason is not None
    assert "row order" in reason


def test_a_likelihood_that_disagrees_with_p_left_fails(tmp_path):
    reason = _violation(tmp_path, _LAPSE_AFTER_P_LEFT)
    assert reason is not None
    assert "p_left" in reason and "likelihood" in reason


def test_a_p_left_without_one_entry_per_trial_fails(tmp_path):
    reason = _violation(tmp_path, _SCALAR_P_LEFT)
    assert reason is not None
    assert "one entry per trial" in reason


def test_a_model_without_p_left_fails(tmp_path):
    reason = _violation(tmp_path, _NO_P_LEFT)
    assert reason is not None
    assert "p_left" in reason


def test_a_potential_on_the_responses_fails(tmp_path):
    reason = _violation(tmp_path, _RESPONSE_POTENTIAL)
    assert reason is not None
    assert "Potential" in reason and "extra" in reason


def test_the_tolerance_is_far_below_a_substantive_lapse():
    # A 1% lapse moves p_left = 0.9 by 0.004; the tolerance is 400x smaller.
    assert CONTRACT_PROBABILITY_TOLERANCE <= 1e-5


def test_admission_rejects_a_contract_breach_before_any_fit(tmp_path):
    candidate = tmp_path / "cand" / "candidate.py"
    candidate.parent.mkdir()
    candidate.write_text(_LAPSE_AFTER_P_LEFT, encoding="utf-8")
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    reason = model_zoo._cheap_gate_rejection(
        candidate, "A hypothesis.", models_dir, "lapse", _write_responses(tmp_path)
    )
    assert reason is not None
    assert reason.startswith("model breaks the data contract")
    assert not (models_dir / "lapse.py").exists()


def test_the_self_check_runs_the_same_contract_check(tmp_path):
    cand_dir = tmp_path / "cand"
    _write_model(cand_dir, "candidate", _FLIPPED)
    with pytest.raises(CandidateCheckFailed, match="model breaks the data contract"):
        run_candidate_check(cand_dir, _write_responses(tmp_path))


def _starting_set(tmp_path: Path, models: dict[str, str]) -> Path:
    models_dir = tmp_path / "models"
    for name, source in models.items():
        _write_model(models_dir, name, source)
    (models_dir / "models_manifest.yaml").write_text(
        yaml.safe_dump({"models": [{"name": n, "rationale": n} for n in models]}),
        encoding="utf-8",
    )
    return models_dir


def test_the_starting_screen_drops_a_carried_breach_on_record(tmp_path):
    models_dir = _starting_set(tmp_path, {"good": _GOOD, "flipped": _FLIPPED})
    ledger = HypothesisLedger.create(tmp_path / "ledger.jsonl", inherit_from=None)
    model_zoo._drop_unfittable_models(
        models_dir,
        _write_responses(tmp_path),
        ledger=ledger,
        ledger_context="experiment 2 start",
        starting_models={"good"},
    )
    assert model_zoo._manifest_names(models_dir) == ["good"]
    (entry,) = ledger.entries()
    assert entry.name == "flipped" and entry.outcome == "dropped"
    assert "data contract" in entry.detail


def test_the_starting_screen_raises_on_a_starting_model_breach(tmp_path):
    models_dir = _starting_set(tmp_path, {"good": _GOOD, "flipped": _FLIPPED})
    with pytest.raises(RuntimeError, match="data contract"):
        model_zoo._drop_unfittable_models(
            models_dir, _write_responses(tmp_path), starting_models={"flipped"}
        )


_PROJECT_SEEDS = (
    REPO_ROOT / "src/pipelines/outer_loop/projects/subjective_randomness/seed_models"
)
_REGISTRY = REPO_ROOT / "src/subjective_randomness/pymc_model_families"
_IMPOSSIBLE = REPO_ROOT / "src/subjective_randomness/impossible_models"


def _model_files():
    """The project's starting models, the ground truths of the faithful sweep
    (the registry manifest's models; the registry's superseded files are
    kept only to refit old runs) and the impossible ground truths."""
    for directory in (_PROJECT_SEEDS, _REGISTRY):
        for entry in read_manifest_entries(directory):
            name = entry["name"]
            yield pytest.param(directory, name, id=f"{directory.name}/{name}")
    for path in sorted(_IMPOSSIBLE.glob("*.py")):
        if path.name != "__init__.py":
            yield pytest.param(_IMPOSSIBLE, path.stem, id=f"{_IMPOSSIBLE.name}/{path.stem}")


@pytest.mark.parametrize("models_dir, name", list(_model_files()))
def test_every_starting_model_and_ground_truth_honours_the_contract(
    tmp_path, models_dir, name
):
    assert model_contract_violation(name, models_dir, _write_responses(tmp_path)) is None
