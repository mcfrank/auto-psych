"""Tiny on-disk inputs for the PyMC inner-loop tests.

Every inner-loop test needs the same three things before it can drive
``run_pymc_inner_loop`` with its expensive seams stubbed: a seed-model
directory with a manifest, a responses CSV, and a canned model posterior. Each
was duplicated verbatim across the inner-loop test modules before it lived here.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def write_seed_models(tmp_path: Path, names=("model_a", "model_b")) -> Path:
    """Create a ``seed_models`` dir holding stub model files and their manifest."""
    seed_dir = tmp_path / "seed_models"
    seed_dir.mkdir()
    for name in names:
        (seed_dir / f"{name}.py").write_text(f"# stub {name}\n", encoding="utf-8")
    (seed_dir / "models_manifest.yaml").write_text(
        yaml.safe_dump(
            {"models": [{"name": name} for name in names]}, sort_keys=False
        ),
        encoding="utf-8",
    )
    return seed_dir


def write_responses(tmp_path: Path) -> Path:
    """Two trials — enough for the loop to have data; never actually fitted.

    Also writes the task description the outer loop puts beside them.
    """
    responses = tmp_path / "responses.csv"
    responses.write_text(
        "sequence_a,sequence_b,participant_id,trial_index,chose_left\n"
        "HHTTHT,HTHTHT,1,0,1\nHHTTHT,HTHTHT,2,0,0\n",
        encoding="utf-8",
    )
    write_task_description_beside(responses)
    return responses


def write_task_description_beside(responses: Path) -> Path:
    """The task description inner-loop agents are told, beside ``responses``."""
    path = Path(responses).parent / "task_description.md"
    path.write_text("# The task\n\nPick the sequence that looks more random.\n",
                    encoding="utf-8")
    return path


def canned_posterior(best: str, others) -> dict:
    """A ``model_posterior`` result putting most of the mass on ``best``.

    Named ``canned_`` rather than ``model_posterior`` because the inner-loop
    tests monkeypatch the production function of that name.
    """
    names = [best] + list(others)
    posteriors = {name: (0.7 if name == best else 0.3 / len(others)) for name in names}
    return {
        "posteriors": posteriors,
        "elpd_loo": {name: -10.0 - i for i, name in enumerate(names)},
        "n_trials": 2,
    }
