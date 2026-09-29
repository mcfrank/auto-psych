"""Agent-facing text describes the raw-only data the agents actually get.

Under the raw-only pipeline ``responses.csv`` carries only ``sequence_a``,
``sequence_b``, ``participant_id``, ``trial_index`` and ``chose_left``; every
model computes its own features (``compute_features`` / ``prepare_observed``)
and critique statistics see the same raw columns. The candidate prompt
(``prompts/pymc_theory.md``) still told agents the CSV held "precomputed feature
columns" and showed a skeleton binding ``n_a`` / ``h_a`` with no hook, and the
critique prompt promised "all feature columns" (first audit, D9).
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pymc as pm
import pytest
import yaml

from src.models.data_binding import make_stim_data
from src.models.model_loading import _COMPUTE_FEATURES_ATTR, load_pymc_model, pm_data_inputs
from src.pipelines.inner_loop import candidate_agent, critique_round
from src.pipelines.outer_loop.columns import RAW_RESPONSE_COLUMNS
from src.subjective_randomness.features import featurize_stimulus
from tests.inner_loop_fixtures import write_task_description_beside
from tests.paths import REPO_ROOT

PROMPTS_DIR = REPO_ROOT / "src" / "pipelines" / "inner_loop" / "prompts"
TASK_DESCRIPTION = (
    REPO_ROOT / "src" / "pipelines" / "outer_loop" / "projects" / "subjective_randomness"
    / "task_description.md"
)
# Every column the retired featurizer used to add to responses.csv.
FEATURE_COLUMNS = sorted(featurize_stimulus("HHTH", "THTT"))
# Phrases that present feature columns as part of the data.
FEATURE_CLAIMS = re.compile(
    r"precomputed feature|all feature columns|feature columns? (?:in|of) the"
    r"|existing (?:feature )?columns|condition on feature columns"
    r"|(?:computed|precomputed) (?:exactly )?by the featurizer",
    re.IGNORECASE,
)
SEED_POOL = (
    REPO_ROOT / "src" / "pipelines" / "outer_loop" / "projects" / "subjective_randomness"
    / "seed_models"
)
FENCED = re.compile(r"```.*?```", re.DOTALL)


def _raw_responses(path: Path) -> Path:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(RAW_RESPONSE_COLUMNS))
        writer.writeheader()
        writer.writerow({"sequence_a": "HHTH", "sequence_b": "THTT", "participant_id": 1,
                         "trial_index": 0, "chose_left": 1})
    write_task_description_beside(path)
    return path


def _candidate_context(tmp_path: Path, responses: Path) -> str:
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    (models_dir / "models_manifest.yaml").write_text(
        yaml.safe_dump({"models": [{"name": "seed", "rationale": "r"}]}), encoding="utf-8"
    )
    (models_dir / "seed.py").write_text("# stub\n", encoding="utf-8")
    candidate_agent._write_candidate_context(
        tmp_path / "candidate0", responses, models_dir,
        iteration=0, candidate_idx=0, candidate_count=1, current_posterior=None,
    )
    return (tmp_path / "candidate0" / "CONTEXT.md").read_text(encoding="utf-8")


def _critique_context(tmp_path: Path, responses: Path) -> str:
    models_dir = tmp_path / "zoo"
    models_dir.mkdir(exist_ok=True)
    (models_dir / "models_manifest.yaml").write_text(
        yaml.safe_dump({"models": [{"name": "incumbent", "rationale": "People do A."}]}),
        encoding="utf-8",
    )
    (models_dir / "incumbent.py").write_text("# stub\n", encoding="utf-8")
    return critique_round._write_critique_context(
        tmp_path / "critique", "incumbent", models_dir, responses, tmp_path / "cache",
        n_proposals=8, significance_alpha=0.05, n_replicates=1000,
    )


@pytest.fixture
def agent_facing_texts(tmp_path):
    responses = _raw_responses(tmp_path / "responses.csv")
    critique_text = _critique_context(tmp_path, responses)
    return {
        "prompts/pymc_theory.md": (PROMPTS_DIR / "pymc_theory.md").read_text(encoding="utf-8"),
        "prompts/critique.md": (PROMPTS_DIR / "critique.md").read_text(encoding="utf-8"),
        "task_description.md": TASK_DESCRIPTION.read_text(encoding="utf-8"),
        "CONTEXT.md": _candidate_context(tmp_path, responses),
        "CRITIQUE_CONTEXT.md": critique_text,
        "critique prompt": critique_round._build_critique_prompt(tmp_path / "critique", critique_text),
        "exploration lenses": "\n".join(candidate_agent.DEFAULT_CANDIDATE_HINTS),
    }


def test_no_agent_facing_text_presents_feature_columns_as_data(agent_facing_texts):
    # The seed models are the worked examples agents read.
    seeds = {path.name: path.read_text(encoding="utf-8") for path in SEED_POOL.glob("*.py")}
    for source, text in {**agent_facing_texts, **seeds}.items():
        assert not FEATURE_CLAIMS.findall(text), (source, FEATURE_CLAIMS.findall(text))


def test_no_agent_facing_text_names_a_feature_column_outside_code_that_computes_it(
    agent_facing_texts,
):
    """A retired feature column may appear only inside an example that
    computes it with ``compute_features``, never as a column of the data."""
    for source, text in agent_facing_texts.items():
        prose = FENCED.sub(
            lambda block: block.group(0) if "def compute_features" not in block.group(0) else "",
            text,
        )
        named = [c for c in FEATURE_COLUMNS if re.search(rf"\b{c}\b", prose)]
        assert not named, (source, named)


def test_the_candidate_prompts_skeleton_computes_its_features_from_the_raw_sequences(tmp_path):
    """The example agents copy binds on raw rows: every pm.Data is either the
    response or a key its compute_features returns."""
    prompt = (PROMPTS_DIR / "pymc_theory.md").read_text(encoding="utf-8")
    skeleton = prompt.split("## Example skeleton", 1)[1]
    code = re.search(r"```python\n(.*?)```", skeleton, re.DOTALL).group(1)
    (tmp_path / "skeleton.py").write_text(code, encoding="utf-8")
    model = load_pymc_model("skeleton", tmp_path)

    rows = [{"sequence_a": "HHTH", "sequence_b": "THTT", "participant_id": 1,
             "trial_index": 0, "chose_left": 1}]
    features = getattr(model, _COMPUTE_FEATURES_ATTR)("HHTH", "THTT")
    assert set(pm_data_inputs(model)) == set(features) | {"chose_left"}
    data = make_stim_data(model, rows)
    with model:
        pm.set_data(data)
    assert np.isfinite(model.compile_logp()(model.initial_point()))


def test_a_brief_for_responses_with_feature_columns_fails_loudly(tmp_path):
    """The brief describes only the raw columns; a CSV with more is not the
    raw-only pipeline's, so writing the brief raises instead of describing it."""
    responses = tmp_path / "responses.csv"
    with responses.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join([*RAW_RESPONSE_COLUMNS, "h_a"]) + "\nHHTH,THTT,1,0,1,3\n")
    write_task_description_beside(responses)
    with pytest.raises(ValueError, match=r"beyond the raw ones"):
        _candidate_context(tmp_path, responses)
