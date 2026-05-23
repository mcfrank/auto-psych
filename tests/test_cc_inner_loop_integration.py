import csv
import json

import yaml

from src.pipelines.outer_loop.orchestrator import run_inner_model_loop_programmatic


def test_inner_model_loop_exports_outer_pipeline_artifacts(tmp_path):
    exp_dir = tmp_path / "project" / "experiment1"
    data_dir = exp_dir / "data"
    data_dir.mkdir(parents=True)
    with (data_dir / "responses.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["participant_id", "trial_index", "sequence_a", "sequence_b", "chose_left"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"participant_id": "p1", "trial_index": 0, "sequence_a": "HTHT", "sequence_b": "HHHH", "chose_left": 1},
                {"participant_id": "p1", "trial_index": 1, "sequence_a": "HHHH", "sequence_b": "HTHT", "chose_left": 0},
            ]
        )

    loop_dir = run_inner_model_loop_programmatic(
        exp_dir,
        max_iterations=0,
        candidate_count=0,
    )

    model_path = exp_dir / "cognitive_models" / "inner_loop_model.py"
    manifest = yaml.safe_load((exp_dir / "cognitive_models" / "models_manifest.yaml").read_text())
    posterior = json.loads((exp_dir / "model_loop" / "model_posterior.json").read_text())

    assert loop_dir == exp_dir / "model_loop"
    assert model_path.exists()
    assert manifest["models"][0]["name"] == "inner_loop_model"
    assert (exp_dir / "model_loop" / "report.md").exists()
    assert "posteriors" in posterior
