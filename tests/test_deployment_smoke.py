import json

from src.pipelines.outer_loop.deployment.smoke import write_smoke_experiment


def test_write_smoke_experiment_creates_deployable_files(tmp_path):
    exp_dir = tmp_path / "experiment1"

    experiment_dir = write_smoke_experiment(exp_dir)

    assert experiment_dir == exp_dir / "experiment"
    assert (experiment_dir / "index.html").exists()
    config = json.loads((experiment_dir / "config.json").read_text(encoding="utf-8"))
    assert config["experiment_url"] is None
    stimuli = json.loads((exp_dir / "design" / "stimuli.json").read_text(encoding="utf-8"))
    assert len(stimuli) >= 1
    assert "jsPsych" in (experiment_dir / "index.html").read_text(encoding="utf-8")
