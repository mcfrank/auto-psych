"""The impossible-GT control array builds its agent tree the way the faithful
array does.

It used to keep the tree at ``run<r>/<gt>/repo`` — so every agent prompt named
the recipe (``more_heads_more_random``) — build it with an old inline exclude
list that kept ``tests/``, ``docs/``, ``CLAUDE.md`` and ``.secrets``, run the
harness from inside the agent tree, pass the agents the harness's whole
environment and never scan for the recipe's name. These tests run the real
array script against a staged harness whose ``_env.sh`` and harness CLI are
stubs that record what they were given.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from tests.paths import REPO_ROOT, load_script_module

SLURM_DIR = REPO_ROOT / "scripts" / "subjective_randomness" / "slurm"
GT = "more_heads_more_random"

# Records the harness's argv and environment, then finishes the cell.
_FAKE_HARNESS = """
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
out = Path(args[args.index("--out") + 1])
Path(os.environ["FAKE_HARNESS_RECORD"]).write_text(json.dumps({"argv": args, "env": dict(os.environ)}))
(Path(args[args.index("--results-root") + 1]) / "cell_1").mkdir(parents=True)
out.write_text("{}")
"""


def _staged_sweep(tmp_path: Path) -> Path:
    work = tmp_path / "work"
    staged = work / "harness_repo" / "scripts" / "subjective_randomness"
    (staged / "slurm").mkdir(parents=True)
    for name in ("cell_lock.sh", "agent_tree.exclude", "scan_gt_name.sh", "agent_activity_report.py"):
        shutil.copy(SLURM_DIR / name, staged / "slurm" / name)
    (staged / "slurm" / "_env.sh").write_text(
        f'export VENV_PY="{sys.executable}"\nexport REPO="{tmp_path / "checkout"}"\n',
        encoding="utf-8",
    )
    (staged / "impossible_holdout_recovery.py").write_text(_FAKE_HARNESS, encoding="utf-8")
    # The staged agent source: what agent_tree.exclude leaves of a checkout.
    agent_src = work / "agent_src"
    (agent_src / "src" / "pipelines").mkdir(parents=True)
    (agent_src / "src" / "pipelines" / "loop.py").write_text("# the loop\n", encoding="utf-8")
    (agent_src / "opencode.json").write_text("{}", encoding="utf-8")
    (work / "code_commit").write_text("abc123\n", encoding="utf-8")
    (work / "impossible_models_src").mkdir()
    (work / "impossible_models_src" / f"{GT}.py").write_text("# the recipe\n", encoding="utf-8")
    (work / "impossible_config_src.yaml").write_text("gt_models_dir: x\n", encoding="utf-8")
    return work


def _run(tmp_path: Path, work: Path) -> subprocess.CompletedProcess:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "squeue").write_text("#!/bin/bash\n", encoding="utf-8")
    (bin_dir / "squeue").chmod(0o755)
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "WORK_ROOT": str(work),
        "GT_MODELS": f"{GT} fewer_heads_more_random",
        "SLURM_ARRAY_TASK_ID": "1",
        "SLURM_JOB_ID": "77",
        "AGENT_TREES_ROOT": str(tmp_path / "agent_trees"),
        "PROLIFIC_API_TOKEN": "prolific",
        "FAKE_HARNESS_RECORD": str(tmp_path / "harness.json"),
    }
    return subprocess.run(
        ["bash", str(SLURM_DIR / "impossible_holdout_recovery_array.sbatch")],
        env=env, capture_output=True, text=True, timeout=120,
    )


def test_the_tree_is_opaque_scrubbed_and_the_harness_runs_outside_it(tmp_path):
    work = _staged_sweep(tmp_path)
    result = _run(tmp_path, work)
    assert result.returncode == 0, result.stdout + result.stderr

    cell = work / "run1" / GT
    tree_id = (cell / "agent_tree_id").read_text(encoding="utf-8")
    assert re.fullmatch(r"[0-9a-f]{16}", tree_id)
    assert GT in result.stdout and "GT-name scan" in result.stdout

    record = json.loads((tmp_path / "harness.json").read_text(encoding="utf-8"))
    argv = record["argv"]
    agent_root = argv[argv.index("--agent-root") + 1]
    assert agent_root == str(tmp_path / "agent_trees" / tree_id / "repo")
    assert GT not in agent_root
    assert argv[argv.index("--summary-root") + 1] == str(work / "run1")
    assert argv[argv.index("--results-root") + 1] == f"{agent_root}/_runs"
    # The harness's environment (the agents' allowlist is taken from it)
    # names neither the recipe nor the sweep.
    env = record["env"]
    assert not {"GT_MODELS", "WORK_ROOT"} & set(env)
    assert not any(GT in value for value in env.values())
    # Finished: archived, tree removed, the leak record written or clean.
    assert (cell / "agent_runs.tar.gz").exists()
    assert not (tmp_path / "agent_trees" / tree_id).exists()


def test_a_tree_that_names_the_recipe_stops_before_any_agent_runs(tmp_path):
    work = _staged_sweep(tmp_path)
    (work / "agent_src" / "src" / "pipelines" / "notes.py").write_text(
        f"# see {GT}\n", encoding="utf-8"
    )
    result = _run(tmp_path, work)
    assert result.returncode != 0
    assert "names the held-out ground truth" in result.stderr
    assert not (tmp_path / "harness.json").exists()


def test_the_impossible_cli_forwards_the_agent_root_and_summary_root(tmp_path, monkeypatch):
    cli = load_script_module(REPO_ROOT / "scripts" / "subjective_randomness" / "impossible_holdout_recovery.py")
    seen = {}

    def fake_run(config, config_path, results_root, **overrides):
        seen.update(overrides)
        return {"gt_runs": []}

    monkeypatch.setattr(cli, "run_impossible_holdout_recovery_from_config", fake_run)
    monkeypatch.setattr(cli, "load_config", lambda path: {})
    cli.main(cli.Args(
        config=tmp_path / "c.yaml", out=tmp_path / "out.json",
        agent_root=tmp_path / "tree", summary_root=tmp_path / "summary",
        agent_model="google/gemini-3.1-pro-preview",
    ))
    assert seen["agent_root"] == tmp_path / "tree"
    assert seen["summary_root"] == tmp_path / "summary"
    assert seen["agent_model_override"] == "google/gemini-3.1-pro-preview"
