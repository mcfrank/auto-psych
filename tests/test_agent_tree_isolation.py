"""Agent tree isolation: the scrubbed copy agents receive contains no feature code.

The rsync exclude file (agent_tree.exclude) is shared between the Slurm array
sbatch and this test, so they always agree on what is excluded.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from pyprojroot import here

REPO = here()
EXCLUDE_FILE = (
    REPO / "scripts" / "subjective_randomness" / "slurm" / "agent_tree.exclude"
)

FORBIDDEN_PATHS = [
    "src/subjective_randomness/features.py",
    "src/subjective_randomness/sequence_stats.py",
    "src/subjective_randomness/stimulus_design.py",
    "src/subjective_randomness/model_recovery.py",
    "src/subjective_randomness/pymc_recover.py",
    "src/subjective_randomness/model_families",
]

FORBIDDEN_GLOBS = [
    "scripts/subjective_randomness/configs/holdout_recovery*.yaml",
]

FORBIDDEN_ANYWHERE = [
    "preprocess.py",
    "ground_truth_models.py",
    "evaluate_recovery.py",
    "gt.txt",
]

FORBIDDEN_DEFS = [
    "def parse_motifs",
    "def featurize_stimulus",
    "def multiscale_local_imbalance",
    "def occurrence_probability",
    "def periodicity_score",
]

SEED_DIRS = [
    "src/pipelines/outer_loop/projects/subjective_randomness/seed_models",
    "src/subjective_randomness/pymc_model_families",
]


def _build_agent_tree(dest: Path) -> None:
    """Rsync the repo into dest using the exclude file, as the sbatch does."""
    subprocess.run(
        [
            "rsync",
            "-a",
            "--exclude-from",
            str(EXCLUDE_FILE),
            str(REPO) + "/",
            str(dest) + "/",
        ],
        check=True,
    )
    (dest / ".here").touch()


@pytest.fixture(scope="module")
def agent_tree(tmp_path_factory):
    tree = tmp_path_factory.mktemp("agent_tree")
    _build_agent_tree(tree)
    return tree


class TestForbiddenPathsAbsent:
    """No feature/research/GT code in the agent tree."""

    @pytest.mark.parametrize("rel", FORBIDDEN_PATHS)
    def test_specific_path_absent(self, agent_tree, rel):
        p = agent_tree / rel
        assert not p.exists(), f"Forbidden path in agent tree: {rel}"

    @pytest.mark.parametrize("name", FORBIDDEN_ANYWHERE)
    def test_filename_absent_everywhere(self, agent_tree, name):
        hits = list(agent_tree.rglob(name))
        assert not hits, f"Forbidden file {name!r} found in agent tree: {hits}"

    def test_holdout_configs_absent(self, agent_tree):
        cfg_dir = agent_tree / "scripts" / "subjective_randomness" / "configs"
        if cfg_dir.exists():
            hits = list(cfg_dir.glob("holdout_recovery*.yaml"))
            assert not hits, f"Holdout config(s) in agent tree: {hits}"


class TestForbiddenDefsOnlyInSeeds:
    """Feature-computing function defs only appear in the visible seed dirs."""

    @pytest.mark.parametrize("pattern", FORBIDDEN_DEFS)
    def test_def_only_in_seeds(self, agent_tree, pattern):
        result = subprocess.run(
            ["grep", "-rn", pattern, str(agent_tree / "src")],
            capture_output=True,
            text=True,
        )
        for line in result.stdout.strip().splitlines():
            if not line:
                continue
            path = line.split(":")[0]
            rel = os.path.relpath(path, agent_tree)
            in_seed = any(rel.startswith(sd) for sd in SEED_DIRS)
            assert in_seed, (
                f"Feature-computing def {pattern!r} found outside seed dirs: {rel}"
            )
