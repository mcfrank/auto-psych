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
    # Alternative implementations of the ground truths (motif_stack_softmax.py
    # describes a held-out motif_stack in full); only opencode honoured the
    # read deny-list that used to cover it.
    "src/subjective_randomness/pymc_model_families",
    # The rest of the research library is harness-only, and several of its
    # modules name the ground truths (incumbent.py, exhaustive_search.py, ...).
    "src/subjective_randomness",
    # Docs, tests, scripts and analyses name and describe the ground truths
    # (40 files named motif_stack); agents need only src/ and their run tree.
    "docs",
    "tests",
    "scripts",
    "analysis",
    "diagrams",
    "README.md",
    "SUMMARY.md",
    "HERO_RUN_DESIDERATA.md",
    # Every agent could read the Prolific, Firebase, Google and Claude tokens.
    ".secrets",
]

GROUND_TRUTHS = [
    "motif_stack",
    "falk_konold_dp",
    "finite_experience_occurrence",
    "local_representativeness",
]
# The seed pool legitimately names the seeds that are NOT held out; the array
# sbatch deletes the held-out one (and its manifest entry) per task.
SEED_POOL = "src/pipelines/outer_loop/projects/subjective_randomness/seed_models"

FORBIDDEN_GLOBS = [
    "scripts/subjective_randomness/configs/holdout_recovery*.yaml",
]

FORBIDDEN_ANYWHERE = [
    "preprocess.py",
    "ground_truth_models.py",
    "evaluate_recovery.py",
    "gt.txt",
    # Agent CLIs load these into every session; the project CLAUDE.md names
    # the held-out model.
    "CLAUDE.md",
    "AGENTS.md",
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
    """Rsync the repo into dest using the exclude file, as the sbatch does.

    --delete-excluded cleans a tree a resumed task built before an exclusion
    was added; the protect filter keeps the agents' results root.
    """
    subprocess.run(
        [
            "rsync",
            "-a",
            "--delete",
            "--delete-excluded",
            "--filter=P /_runs/***",
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


class TestNoGroundTruthNamed:
    """Beyond the seed pool, no file in the agent tree names a ground truth.

    A loader comment ("e.g. motif_stack's unique-sequence table") reached three
    agents, and a path naming the held-out model once led an agent to build its
    first model from the name.
    """

    @pytest.mark.parametrize("gt", GROUND_TRUTHS)
    def test_ground_truth_named_only_in_the_seed_pool(self, agent_tree, gt):
        result = subprocess.run(
            ["grep", "-rIl", "--", gt, str(agent_tree)],
            capture_output=True, text=True,
        )
        hits = [
            os.path.relpath(line, agent_tree)
            for line in result.stdout.splitlines()
            if not os.path.relpath(line, agent_tree).startswith(SEED_POOL)
        ]
        assert not hits, f"{gt!r} named in the agent tree: {hits}"


def test_rebuilding_a_resumed_tree_removes_newly_excluded_files_but_keeps_runs(tmp_path):
    tree = tmp_path / "repo"
    (tree / "docs").mkdir(parents=True)
    (tree / "docs" / "old.md").write_text("motif_stack\n", encoding="utf-8")
    (tree / ".secrets").write_text("TOKEN=x\n", encoding="utf-8")
    # The results root holds names the exclude list matches ("data", "*.nc");
    # all of it must survive the rebuild.
    kept = [
        tree / "_runs" / "cell_1" / "history.json",
        tree / "_runs" / "cell_1" / "experiment1" / "data" / "responses.csv",
        tree / "_runs" / "cell_1" / "fits" / "model.abc.nc",
    ]
    for path in kept:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("kept", encoding="utf-8")
    _build_agent_tree(tree)
    assert not (tree / "docs").exists()
    assert not (tree / ".secrets").exists()
    assert all(path.read_text() == "kept" for path in kept)
