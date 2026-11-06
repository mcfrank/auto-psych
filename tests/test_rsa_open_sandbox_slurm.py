"""An agent in the sandbox cannot reach Slurm (review V3 §11, probed on Sherlock 2026-11-05).

The sandbox mounts /usr (so the Slurm binaries are visible) but not /run, where Slurm's
authentication socket lives, and passes no ``SLURM_*`` variable; ``squeue`` then fails to
authenticate. That is a property of the mounts, not a guarantee: this test fails if a change to
the sandbox (or to Slurm) lets an agent's shell talk to the scheduler. It runs only where both
bubblewrap and Slurm are installed (a Sherlock compute node); elsewhere it is skipped."""

import os
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.skipif(not (shutil.which("bwrap") and shutil.which("squeue")),
                                reason="needs bubblewrap and Slurm (a cluster node)")


def test_squeue_works_outside_the_sandbox_and_fails_inside_it(tmp_path):
    from src.runtime.agent_sandbox import sandbox_command

    outside = subprocess.run(["squeue", "--me", "-h"], capture_output=True, text=True, timeout=60)
    assert outside.returncode == 0, outside.stderr  # Slurm works here, so a failure inside means something
    tree, agent = tmp_path / "tree", tmp_path / "agent"
    tree.mkdir()
    agent.mkdir()
    cmd, env = sandbox_command(["squeue", "--me", "-h"], backend="opencode", cwd=tree, writable_dirs=[],
                               agent_dir=agent, env=dict(os.environ))
    assert not [k for k in env if k.startswith("SLURM_")]
    inside = subprocess.run(cmd, env=env, cwd=tree, capture_output=True, text=True, timeout=60)
    assert inside.returncode != 0, ("an agent's shell reached Slurm", inside.stdout, inside.stderr)
