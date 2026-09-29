"""scan_gt_name.sh: the held-out model's name in an agent tree or in agent output.

Before agents start (--before-agents), a name in the tree is our leak: the
cell stops, having spent nothing. At the end of a run (--after-run), a name in
what the agents wrote may be a coincidence (an agent can coin "motif_stack"
unaided): the scan lists the files and warns, and never fails the cell.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from pyprojroot import here

SCAN = here() / "scripts" / "subjective_randomness" / "slurm" / "scan_gt_name.sh"


def _scan(mode, gt, directory, out):
    return subprocess.run(
        ["bash", str(SCAN), mode, gt, str(directory), str(out)],
        capture_output=True, text=True,
    )


def test_a_tree_naming_the_ground_truth_stops_the_cell_before_agents(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "loader.py").write_text("# e.g. motif_stack's table\n")
    result = _scan("--before-agents", "motif_stack", tmp_path, tmp_path / "m.txt")
    assert result.returncode != 0
    assert "src/loader.py" in result.stderr


def test_a_clean_tree_passes(tmp_path):
    (tmp_path / "loader.py").write_text("# a per-sequence table\n")
    result = _scan("--before-agents", "motif_stack", tmp_path, tmp_path / "m.txt")
    assert result.returncode == 0, result.stderr


def test_agent_output_naming_the_ground_truth_warns_and_never_fails(tmp_path):
    runs = tmp_path / "_runs"
    (runs / "candidate_0").mkdir(parents=True)
    (runs / "candidate_0" / "hypothesis.md").write_text("A motif_stack of chunks.\n")
    out = tmp_path / "gt_name_mentions.txt"
    result = _scan("--after-run", "motif_stack", runs, out)
    assert result.returncode == 0
    assert "WARNING" in result.stdout
    assert out.read_text().strip().endswith("candidate_0/hypothesis.md")


def test_clean_agent_output_leaves_no_mentions_file(tmp_path):
    (tmp_path / "hypothesis.md").write_text("Mirror symmetry.\n")
    out = tmp_path / "gt_name_mentions.txt"
    result = _scan("--after-run", "motif_stack", tmp_path, out)
    assert result.returncode == 0
    assert not out.exists()
