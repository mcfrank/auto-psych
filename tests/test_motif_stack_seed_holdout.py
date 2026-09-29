"""Holding motif_stack out withholds the motif_stack seed, and nothing
resembling the motif-stack model reaches the agents.

The seed pool's ``motif_stack.py`` is the registry's Viterbi model, byte for
byte (on 2026-09-27 it was briefly the softmax rewrite
``motif_stack_softmax.py``, reverted because it failed the convergence gate on
two of the three ground truths' data). Everything that withholds the held-out
ground truth works by name: the array deletes ``<gt>.py`` and every ``*<gt>*``
file from the agent tree and scrubs ``<gt>`` from the manifests, the harness
withholds the seed whose manifest name is the ground truth, and the name scan
stops a cell whose tree names it.

The array tests run the real ``holdout_recovery_array.sbatch`` against a staged
harness whose ``_env.sh`` and harness CLI are stubs, keep the agent tree
(``KEEP_REPO_COPY``) and inspect it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from src.models.model_manifest import read_manifest_names
from src.pipelines.outer_loop.orchestrator import (
    project_seed_models_dir,
    seed_experiment_models_from_project,
)
from src.subjective_randomness.holdout_data import seed_exclusion
from tests.paths import REPO_ROOT

SLURM_DIR = REPO_ROOT / "scripts" / "subjective_randomness" / "slurm"
REGISTRY_DIR = REPO_ROOT / "src" / "subjective_randomness" / "pymc_model_families"
FAMILY_DIR = REPO_ROOT / "src" / "subjective_randomness" / "model_families"
POOL_REL = "src/pipelines/outer_loop/projects/subjective_randomness/seed_models"
VITERBI = (REGISTRY_DIR / "motif_stack.py").read_bytes()
# Code of the motif-stack model that no other seed has: the automaton's
# transition matrices, its memory-method flags, its regular likelihood.
MOTIF_STACK_CODE = ["def _matrices(", "def _memory_flags(", "def _log_p_regular("]

# Finishes the cell without running anything.
_FAKE_HARNESS = """
import sys
from pathlib import Path
args = sys.argv[1:]
Path(args[args.index("--out") + 1]).write_text("{}")
"""


@pytest.fixture(scope="module")
def staged_sweep(tmp_path_factory):
    """harness_repo (the real slurm helpers, a stub _env.sh and harness) and
    agent_src (the repo through agent_tree.exclude), as the setup job stages
    them."""
    work = tmp_path_factory.mktemp("staged")
    slurm = work / "harness_repo" / "scripts" / "subjective_randomness" / "slurm"
    slurm.mkdir(parents=True)
    for name in ("cell_lock.sh", "agent_tree.exclude", "scan_gt_name.sh"):
        shutil.copy(SLURM_DIR / name, slurm / name)
    (slurm / "_env.sh").write_text(
        f'export VENV_PY="{sys.executable}"\nexport REPO="{work / "checkout"}"\n',
        encoding="utf-8",
    )
    scripts = slurm.parent
    shutil.copy(REPO_ROOT / "scripts" / "subjective_randomness" / "remove_manifest_entry.py",
                scripts / "remove_manifest_entry.py")
    (scripts / "holdout_recovery.py").write_text(_FAKE_HARNESS, encoding="utf-8")
    subprocess.run(
        ["rsync", "-a", "--exclude-from", str(SLURM_DIR / "agent_tree.exclude"),
         f"{REPO_ROOT}/", f"{work / 'agent_src'}/"],
        check=True,
    )
    shutil.copytree(REGISTRY_DIR, work / "gt_models_src")
    shutil.copytree(FAMILY_DIR, work / "gt_family_src")
    return work


def _agent_tree(tmp_path: Path, staged: Path, gt: str) -> Path:
    """Run the array for ``gt`` and return the agent tree it built."""
    work = tmp_path / "work"
    work.mkdir()
    (work / "code_commit").write_text("abc123\n", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "squeue").write_text("#!/bin/bash\n", encoding="utf-8")
    (bin_dir / "squeue").chmod(0o755)
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "WORK_ROOT": str(work),
        "HARNESS_REPO": str(staged / "harness_repo"),
        "AGENT_SRC": str(staged / "agent_src"),
        "GT_MODELS_SRC": str(staged / "gt_models_src"),
        "GT_FAMILY_SRC": str(staged / "gt_family_src"),
        "GT_MODELS": gt,
        "SLURM_ARRAY_TASK_ID": "1",
        "SLURM_JOB_ID": "77",
        "AGENT_TREES_ROOT": str(tmp_path / "agent_trees"),
        "KEEP_REPO_COPY": "1",
    }
    result = subprocess.run(
        ["bash", str(SLURM_DIR / "holdout_recovery_array.sbatch")],
        env=env, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    tree_id = (work / "run1" / gt / "agent_tree_id").read_text(encoding="utf-8")
    return tmp_path / "agent_trees" / tree_id / "repo"


def _tree_files(tree: Path) -> list[Path]:
    return [
        path for path in tree.rglob("*")
        if path.is_file() and not path.is_symlink() and "_runs" not in path.relative_to(tree).parts
    ]


# (This test's name must not contain the ground truth's: it is part of
# tmp_path, and the array refuses an agent directory that names the held-out
# model.)
def test_holding_out_the_motif_model_leaves_nothing_resembling_it_in_the_agent_tree(
    tmp_path, staged_sweep
):
    tree = _agent_tree(tmp_path, staged_sweep, "motif_stack")
    files = _tree_files(tree)
    assert (tree / POOL_REL / "models_manifest.yaml").exists()

    # By name: no file, and no manifest entry.
    assert not [p for p in files if "motif_stack" in p.name]
    for manifest in tree.rglob("models_manifest.yaml"):
        assert "motif_stack" not in manifest.read_text(encoding="utf-8"), manifest
        assert not [n for n in read_manifest_names(manifest.parent) if "motif" in n]

    # By content: the motif-stack file under no name, and none of its code.
    for path in files:
        data = path.read_bytes()
        assert data != VITERBI, path
        text = data.decode("utf-8", errors="replace")
        assert not [code for code in MOTIF_STACK_CODE if code in text], path


def test_with_another_ground_truth_held_out_the_agents_motif_stack_seed_is_the_viterbi_model(
    tmp_path, staged_sweep
):
    tree = _agent_tree(tmp_path, staged_sweep, "falk_konold_dp")
    pool = tree / POOL_REL
    assert "motif_stack" in read_manifest_names(pool)
    assert "falk_konold_dp" not in read_manifest_names(pool)
    assert (pool / "motif_stack.py").read_bytes() == VITERBI


@pytest.mark.parametrize(
    "gt, seeded_motif_stack",
    [("falk_konold_dp", VITERBI), ("local_representativeness", VITERBI), ("motif_stack", None)],
)
def test_the_harness_seeds_experiment_1_with_the_viterbi_motif_stack_unless_it_is_held_out(
    tmp_path, gt, seeded_motif_stack
):
    pool = project_seed_models_dir("subjective_randomness")
    exp_dir = tmp_path / "experiment1"
    assert seed_experiment_models_from_project(
        exp_dir, "subjective_randomness", exclude=seed_exclusion(gt, pool)
    )
    seeded = exp_dir / "cognitive_models"
    names = read_manifest_names(seeded)
    assert gt not in names
    if seeded_motif_stack is None:
        assert not list(seeded.glob("*motif_stack*"))
        assert not [
            p for p in seeded.glob("*.py")
            if any(code in p.read_text(encoding="utf-8") for code in MOTIF_STACK_CODE)
        ]
    else:
        assert (seeded / "motif_stack.py").read_bytes() == seeded_motif_stack

