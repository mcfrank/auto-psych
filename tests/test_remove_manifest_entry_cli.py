"""The holdout sandbox scrub CLI: remove one model from a models_manifest.yaml.

The holdout recovery array task deletes the held-out model's ``.py`` from the
agents' repo copy and then runs this script on both manifests the agents can
open (the GT registry and the live seed pool), so neither names nor describes
the held-out model. The library function is tested in test_model_manifest.py;
these tests pin the script's contract: it removes the entry, it verifies the
name is gone afterwards, and an absent name is an error unless the caller says
a superseded (never-listed) ground truth is expected.
"""

from __future__ import annotations

import pytest
import yaml

from src.models.model_manifest import MANIFEST_FILENAME, read_manifest_entries
from tests.paths import REPO_ROOT, load_script_module

SCRIPT = REPO_ROOT / "scripts" / "subjective_randomness" / "remove_manifest_entry.py"


def _load():
    return load_script_module(SCRIPT, "_sr_script_remove_manifest_entry")


def _manifest(tmp_path, names):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / MANIFEST_FILENAME).write_text(
        "# active: " + ", ".join(names) + "\n"
        + yaml.safe_dump(
            {"models": [{"name": n, "rationale": f"mechanism of {n}"} for n in names]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return models_dir


def test_cli_removes_the_entry_and_reports_it(tmp_path, capsys):
    mod = _load()
    models_dir = _manifest(tmp_path, ["keep_a", "held_out", "keep_b"])

    mod.main(mod.Args(models_dir=models_dir, name="held_out"))

    assert [e["name"] for e in read_manifest_entries(models_dir)] == ["keep_a", "keep_b"]
    assert "held_out" not in (models_dir / MANIFEST_FILENAME).read_text(encoding="utf-8")
    assert "removed 'held_out'" in capsys.readouterr().out


def test_cli_fails_loudly_when_the_name_is_not_listed(tmp_path):
    mod = _load()
    models_dir = _manifest(tmp_path, ["keep_a"])

    with pytest.raises(SystemExit, match="not_listed"):
        mod.main(mod.Args(models_dir=models_dir, name="not_listed"))


def test_cli_tolerates_an_absent_name_only_when_asked(tmp_path, capsys):
    mod = _load()
    models_dir = _manifest(tmp_path, ["keep_a"])
    before = (models_dir / MANIFEST_FILENAME).read_text(encoding="utf-8")

    mod.main(mod.Args(models_dir=models_dir, name="not_listed", missing_ok=True))

    assert (models_dir / MANIFEST_FILENAME).read_text(encoding="utf-8") == before
    assert "not listed" in capsys.readouterr().out


def test_cli_defaults_missing_ok_to_false():
    mod = _load()
    args = mod.Args(models_dir=SCRIPT.parent, name="x")
    assert args.missing_ok is False
