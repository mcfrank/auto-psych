"""The display contents behind panels.json (scripts/paper/rsa_panel_displays.py)."""

import hashlib
import json

import pandas as pd
import pytest

from scripts.paper import rsa_panel_displays


@pytest.fixture
def inputs(tmp_path):
    """Tiny pooled and prereg trial files, and a panels.json that matches them."""
    from src.rsa.dataset import DEFAULT_TRIALS_CSV

    df = pd.read_csv(DEFAULT_TRIALS_CSV)
    pooled = tmp_path / "pooled.csv"
    df[df.experiment == "E8_levels"].assign(source="pragmods").to_csv(pooled, index=False)
    prereg = tmp_path / "prereg.csv"
    df[df.experiment == "E9_twins"].assign(source="auto_psych").to_csv(prereg, index=False)
    entries = rsa_panel_displays.group_displays(pooled, prereg)
    panels_doc = dict(
        pooled_sha256=hashlib.sha256(pooled.read_bytes()).hexdigest(),
        prereg_sha256=hashlib.sha256(prereg.read_bytes()).hexdigest(),
        panels=[dict(dataset=e["dataset"], source=e["source"], experiment=e["experiment"], n=e["n"],
                     classes=[dict(cls=int(c), count=n) for c, n in e["class_counts"].items()])
                for e in entries])
    panels = tmp_path / "panels.json"
    panels.write_text(json.dumps(panels_doc))
    return dict(df=df, pooled=pooled, prereg=prereg, panels=panels, panels_doc=panels_doc, entries=entries)


def test_writes_one_aligned_entry_per_display(inputs, tmp_path):
    out = tmp_path / "panel_displays.json"
    rsa_panel_displays.main(rsa_panel_displays.Args(
        pooled=inputs["pooled"], prereg=inputs["prereg"], panels=inputs["panels"], out=out))
    written = json.loads(out.read_text())
    assert {e["dataset"] for e in written} == {"literature", "prereg"}
    df = inputs["df"]
    n_trials = len(df[df.experiment.isin(["E8_levels", "E9_twins"]) & df.included.astype(bool)
                      & (df.dv == "forced_choice") & (df.objects != "[]")])
    assert sum(e["n"] for e in written) == n_trials
    for e in written:
        assert sum(e["class_counts"].values()) == e["n"]
        assert len(e["objects"]) == len(e["choice_classes"])
        assert all(len(row) == len(e["feature_names"]) for row in e["objects"])
    assert "participant" not in out.read_text()


def test_a_changed_input_file_is_refused(inputs):
    inputs["panels_doc"]["prereg_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sha256"):
        rsa_panel_displays.check_alignment(inputs["entries"], inputs["panels_doc"], inputs["pooled"],
                                           inputs["prereg"])


def test_a_count_or_order_mismatch_is_refused(inputs):
    doc = inputs["panels_doc"]
    doc["panels"][0]["classes"][0]["count"] += 1
    with pytest.raises(ValueError, match="class counts"):
        rsa_panel_displays.check_alignment(inputs["entries"], doc, inputs["pooled"], inputs["prereg"])
    doc["panels"][0]["classes"][0]["count"] -= 1
    doc["panels"] = doc["panels"][::-1]
    with pytest.raises(ValueError, match="display 0"):
        rsa_panel_displays.check_alignment(inputs["entries"], doc, inputs["pooled"], inputs["prereg"])
    doc["panels"] = doc["panels"][1:]
    with pytest.raises(ValueError, match="displays, but panels.json has"):
        rsa_panel_displays.check_alignment(inputs["entries"], doc, inputs["pooled"], inputs["prereg"])
