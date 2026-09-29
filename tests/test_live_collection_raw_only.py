"""Collected live data reaches the agents as the five raw columns only.

``/results`` (``functions/index.js``) returns eight columns, among them
``participant_id_str`` (the Prolific ID) and ``chose_right``/``model``.
Collection wrote them all to ``data/responses.csv`` and pooling carried them
into ``model_loop/responses.csv``, where the first candidate round raised on
them — after the participants were paid — and the critique agent could read
the Prolific IDs first. The raw collected file is kept for the researchers
beside the experiment directories, where no agent is given access.
"""

from __future__ import annotations

import io

import yaml

from src.pipelines.outer_loop import collect
from src.pipelines.outer_loop import model_loop_runner as mlr
from src.pipelines.outer_loop import orchestrator
from src.pipelines.outer_loop.columns import RAW_RESPONSE_COLUMNS

PROJECT = "subjective_randomness"
PROLIFIC_ID = "5f8a1b2c3d4e5f6a7b8c9d0e"
OTHER_PROLIFIC_ID = "60a1b2c3d4e5f6a7b8c9d0e1"
EXTRA_COLUMNS = ("participant_id_str", "chose_right", "model")

# What /results returns: responsesToCsv's columns, in its order.
RESULTS_CSV = "\n".join(
    [
        "participant_id,participant_id_str,trial_index,sequence_a,sequence_b,chose_left,chose_right,model",
        f"0,{PROLIFIC_ID},0,HHT,THT,1,0,",
        f"0,{PROLIFIC_ID},1,HTHT,HHHH,0,1,",
        f"1,{OTHER_PROLIFIC_ID},0,HHT,THT,0,1,",
        f"1,{OTHER_PROLIFIC_ID},1,HTHT,HHHH,1,0,",
    ]
)


def _collect_live_results(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTO_PSYCH_OUTPUT_DIR", str(tmp_path / "output"))
    exp_dir = orchestrator.experiment_dir(PROJECT, 1)
    (exp_dir / "experiment").mkdir(parents=True)
    (exp_dir / "experiment" / "config.json").write_text(
        '{"prolific_study_id": "study1", "results_api_url": "http://results.invalid",'
        ' "total_available_places": 2}',
        encoding="utf-8",
    )
    monkeypatch.setattr(collect, "_poll_prolific_until_target", lambda *a, **k: 2)
    monkeypatch.setattr(
        collect.urllib.request,
        "urlopen",
        lambda *a, **k: io.BytesIO(RESULTS_CSV.encode("utf-8")),
    )
    orchestrator.run_collect_programmatic(
        exp_dir, mode="live", n_participants=2, project_id=PROJECT, prolific_mode="live"
    )
    return exp_dir


def _prepare_model_loop(exp_dir, monkeypatch):
    """Run 5_model_loop's data preparation; the stand-in loop writes a real
    candidate context from the pooled responses, as the first round does."""
    from src.pipelines.inner_loop.candidate_agent import _write_candidate_context

    models_dir = exp_dir / "cognitive_models"
    models_dir.mkdir()
    (models_dir / "models_manifest.yaml").write_text(
        yaml.safe_dump({"models": [{"name": "falk_konold_dp", "rationale": "seed"}]}),
        encoding="utf-8",
    )
    (models_dir / "falk_konold_dp.py").write_text("# stub\n", encoding="utf-8")

    def first_round(responses_path, loop_dir, **kw):
        _write_candidate_context(
            loop_dir / "iter_0" / "candidate_0",
            responses_path,
            models_dir,
            iteration=0,
            candidate_idx=0,
            candidate_count=1,
            current_posterior=None,
        )
        return {"best_model": "falk_konold_dp"}

    monkeypatch.setattr(
        "src.pipelines.inner_loop.pymc_orchestrator.run_pymc_inner_loop", first_round
    )
    monkeypatch.setattr(mlr, "_export_inner_loop_models", lambda e, l, *, best_model: e)
    mlr.run_inner_model_loop_programmatic(
        exp_dir, max_iterations=1, candidate_count=1, project_id=PROJECT
    )


def test_agents_see_only_raw_columns_and_no_prolific_id(tmp_path, monkeypatch):
    exp_dir = _collect_live_results(tmp_path, monkeypatch)
    _prepare_model_loop(exp_dir, monkeypatch)

    # Every file an agent can be given lives under the experiment directory
    # (3_implement: experiment<N>/; loop agents: model_loop/ and the zoo).
    agent_visible = [path for path in exp_dir.rglob("*") if path.is_file()]
    assert exp_dir / "data" / "responses.csv" in agent_visible
    assert exp_dir / "model_loop" / "responses.csv" in agent_visible
    for path in agent_visible:
        text = path.read_text(encoding="utf-8")
        assert PROLIFIC_ID not in text and OTHER_PROLIFIC_ID not in text, path
        for column in EXTRA_COLUMNS:
            assert column not in text.splitlines()[0].split(","), (path, column)
    for name in ("data/responses.csv", "model_loop/responses.csv"):
        header = (exp_dir / name).read_text(encoding="utf-8").splitlines()[0]
        assert header.split(",") == list(RAW_RESPONSE_COLUMNS)
    assert (exp_dir / "model_loop" / "responses.csv").read_text(encoding="utf-8").count(
        "\n"
    ) == 5


def test_the_researchers_keep_the_raw_collected_file_outside_the_experiment(
    tmp_path, monkeypatch
):
    exp_dir = _collect_live_results(tmp_path, monkeypatch)

    raw = orchestrator.raw_collected_responses_path(exp_dir)
    assert exp_dir not in raw.parents
    assert raw.read_text(encoding="utf-8").strip() == RESULTS_CSV.strip()


def test_pooling_keeps_only_raw_columns_from_an_older_wide_file(tmp_path, monkeypatch):
    """A run collected before the fix and resumed: its data/responses.csv
    still has every /results column; the loop's copy must not."""
    monkeypatch.setenv("AUTO_PSYCH_OUTPUT_DIR", str(tmp_path / "output"))
    exp_dir = orchestrator.experiment_dir(PROJECT, 1)
    (exp_dir / "data").mkdir(parents=True)
    (exp_dir / "data" / "responses.csv").write_text(
        RESULTS_CSV + "\n", encoding="utf-8"
    )

    rows = mlr._pooled_response_rows(exp_dir)
    assert [list(row) for row in rows] == [list(RAW_RESPONSE_COLUMNS)] * 4
