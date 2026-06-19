import pytest

import src.runtime.prolific as prol


def test_load_prolific_config_raises_on_malformed_yaml(tmp_path, monkeypatch):
    proj = tmp_path / "badproj"
    proj.mkdir()
    (proj / "prolific_config.yaml").write_text("reward: [unclosed\n", encoding="utf-8")
    monkeypatch.setattr(prol, "project_dir", lambda pid: tmp_path / pid)

    with pytest.raises(ValueError, match="Malformed Prolific config"):
        prol.load_prolific_config("badproj")


def test_load_prolific_config_overrides_defaults(tmp_path, monkeypatch):
    proj = tmp_path / "goodproj"
    proj.mkdir()
    (proj / "prolific_config.yaml").write_text(
        "reward_per_hour: 1200\nestimated_completion_time: 5\n", encoding="utf-8"
    )
    monkeypatch.setattr(prol, "project_dir", lambda pid: tmp_path / pid)

    cfg = prol.load_prolific_config("goodproj")
    assert cfg["reward_per_hour"] == 1200
    assert cfg["estimated_completion_time"] == 5
