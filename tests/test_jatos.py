"""Tests for JATOS config loading, result ZIP parsing, and study archive build."""

import io
import json
import zipfile

import pytest

from src.jatos import parse_jatos_results_zip
from src.jatos_build import build_jzip, get_batch_and_component_ids
from src.jatos_config import load_jatos_config


def test_load_jatos_config_returns_all_keys():
    """load_jatos_config returns dict with expected keys (values may be None)."""
    config = load_jatos_config()
    assert "jatos_base_url" in config
    assert "jatos_study_run_url" in config
    assert "jatos_component_id" in config
    assert "jatos_api_token" in config


def test_parse_jatos_results_zip_extracts_trials():
    """Parse a minimal JATOS result ZIP into responses.csv-style rows."""
    trials = [
        {"sequence_a": "HHTHTTHT", "sequence_b": "THTHTHTH", "chose_left": True},
        {"sequence_a": "TTTTTTTT", "sequence_b": "HHHHHHHH", "chose_left": False},
    ]
    data_txt = json.dumps(trials)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("study_result_1/comp_result_1/data.txt", data_txt)
    zip_bytes = buf.getvalue()

    rows = parse_jatos_results_zip(zip_bytes)
    assert len(rows) == 2
    assert rows[0]["participant_id"] == 0
    assert rows[0]["trial_index"] == 0
    assert rows[0]["sequence_a"] == "HHTHTTHT"
    assert rows[0]["sequence_b"] == "THTHTHTH"
    assert rows[0]["chose_left"] == 1
    assert rows[0]["chose_right"] == 0
    assert rows[1]["chose_left"] == 0
    assert rows[1]["chose_right"] == 1


def test_build_jzip_produces_valid_archive(tmp_path):
    """build_jzip creates a ZIP with one .jas file and study_assets/index.html."""
    (tmp_path / "index.html").write_text("<html><body>Test</body></html>")
    jzip_bytes = build_jzip(tmp_path, study_title="Test Study")
    assert len(jzip_bytes) > 0
    with zipfile.ZipFile(io.BytesIO(jzip_bytes), "r") as zf:
        names = zf.namelist()
        jas_names = [n for n in names if n.endswith(".jas")]
        assert len(jas_names) == 1
        assert any("study_assets/index.html" in n for n in names)
        jas_content = zf.read(jas_names[0]).decode("utf-8")
        jas = json.loads(jas_content)
        assert jas.get("version") == "3"
        assert jas.get("data", {}).get("title") == "Test Study"
        assert len(jas.get("data", {}).get("componentList", [])) == 1
        assert len(jas.get("data", {}).get("batchList", [])) == 1


def test_get_batch_and_component_ids():
    """Extract batch and component IDs from properties (top-level or under data)."""
    batch_id, comp_id = get_batch_and_component_ids({
        "batchList": [{"id": 10}],
        "componentList": [{"id": 20}],
    })
    assert batch_id == 10
    assert comp_id == 20
    batch_id2, comp_id2 = get_batch_and_component_ids({
        "data": {"batchList": [{"id": 5}], "componentList": [{"id": 6}]},
    })
    assert batch_id2 == 5
    assert comp_id2 == 6


def test_parse_jatos_results_zip_skips_non_judgment_trials():
    """Only trials with sequence_a, sequence_b, chose_left are included."""
    data = [
        {"sequence_a": "HH", "sequence_b": "TT", "chose_left": True},
        {"trial_type": "instructions"},
        {"sequence_a": "HT", "sequence_b": "TH", "chose_left": False},
    ]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("study_result_1/comp_result_1/data.txt", json.dumps(data))
    rows = parse_jatos_results_zip(buf.getvalue())
    assert len(rows) == 2
