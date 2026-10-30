"""Every environment script of a job that launches loop agents loads bubblewrap.

Every loop agent (3_implement, critique, candidates) runs sandboxed, and
``agent_sandbox.sandbox_command`` raises when ``bwrap`` is not on PATH;
Sherlock's nodes have no system ``bwrap``, only the ``system bubblewrap``
module. The simulation ``_env.sh`` loaded it; the live one did ``ml purge``
and never did, so a live run would have stopped at its first agent. Each
script is sourced here under a stand-in ``ml`` that records what it loads
and provides ``bwrap`` only for ``ml load system bubblewrap``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from tests.paths import REPO_ROOT

# The env scripts sourced by the jobs that run the pipeline's agents: the live
# runs (setup.sbatch, run_live.sbatch, submit_parallel.sh) and the holdout
# sweeps (setup, array, retry and analysis sbatch files).
AGENT_JOB_ENV_SCRIPTS = [
    REPO_ROOT / "scripts" / "outer_loop_live" / "_env.sh",
    REPO_ROOT / "scripts" / "subjective_randomness" / "slurm" / "_env.sh",
    # The RSA sweep's (it sources the one above).
    REPO_ROOT / "scripts" / "rsa" / "slurm" / "_env.sh",
]
# What the scripts run besides shell builtins and ml.
TOOLS = ("bash", "mkdir", "dirname", "basename", "xargs", "tr", "cat", "echo", "chmod")

_STAND_IN_ML = """#!/bin/bash
echo "$*" >> "$ML_LOG"
if [[ "$*" == "load system bubblewrap" && -n "$MODULE_PROVIDES_BWRAP" ]]; then
  printf '#!/bin/sh\\n' > "$MODULE_BIN/bwrap"
  chmod +x "$MODULE_BIN/bwrap"
fi
"""


def _source(env_script: Path, tmp_path: Path, *, module_provides_bwrap: bool):
    tools = tmp_path / "tools"
    module_bin = tmp_path / "module_bin"
    for directory in (tools, module_bin, tmp_path / "repo"):
        directory.mkdir(exist_ok=True)
    for tool in TOOLS:
        (tools / tool).symlink_to(shutil.which(tool))
    (tools / "ml").write_text(_STAND_IN_ML, encoding="utf-8")
    (tools / "ml").chmod(0o755)
    ml_log = tmp_path / "ml.log"
    result = subprocess.run(
        [
            str(tools / "bash"),
            "-c",
            'source "$1" >/dev/null && command -v bwrap',
            "_",
            str(env_script),
        ],
        env={
            "PATH": f"{module_bin}:{tools}",
            "HOME": str(tmp_path),
            "REPO": str(tmp_path / "repo"),
            "WORK_ROOT": str(tmp_path / "work"),
            "ML_LOG": str(ml_log),
            "MODULE_BIN": str(module_bin),
            "MODULE_PROVIDES_BWRAP": "1" if module_provides_bwrap else "",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    loads = ml_log.read_text(encoding="utf-8").splitlines() if ml_log.exists() else []
    return result, loads


@pytest.mark.parametrize(
    "env_script", AGENT_JOB_ENV_SCRIPTS, ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_the_env_script_puts_bwrap_on_path(env_script, tmp_path):
    result, loads = _source(env_script, tmp_path, module_provides_bwrap=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(tmp_path / "module_bin" / "bwrap")
    assert "purge" in loads
    assert loads.index("load system bubblewrap") > loads.index("purge")


@pytest.mark.parametrize(
    "env_script", AGENT_JOB_ENV_SCRIPTS, ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_the_env_script_stops_when_bubblewrap_is_unavailable(env_script, tmp_path):
    result, _ = _source(env_script, tmp_path, module_provides_bwrap=False)
    assert result.returncode != 0
    assert "FATAL: bwrap not on PATH" in result.stderr


def test_on_coco_bwrap_and_uv_come_from_the_shared_home(tmp_path):
    # /scr is node-local: a tool kept there exists only on the nodes it was
    # installed on, so a cell on any other node would stop.
    home_bin = tmp_path / ".local" / "bin"
    home_bin.mkdir(parents=True)
    for tool in ("bwrap", "uv"):
        (home_bin / tool).write_text("#!/bin/sh\n", encoding="utf-8")
        (home_bin / tool).chmod(0o755)
    tools = tmp_path / "tools"
    tools.mkdir()
    for tool in TOOLS:
        (tools / tool).symlink_to(shutil.which(tool))
    result = subprocess.run(
        [str(tools / "bash"), "-c", 'source "$1" >/dev/null && command -v bwrap && command -v uv && echo "$PATH"', "_",
         str(REPO_ROOT / "scripts" / "subjective_randomness" / "slurm" / "_env.sh")],
        env={"PATH": str(tools), "HOME": str(tmp_path), "USER": "me", "CLUSTER": "coco",
             "REPO": str(tmp_path / "repo"), "WORK_ROOT": str(tmp_path / "work"),
             "SCRATCH": str(tmp_path / "scr"), "L_SCRATCH": str(tmp_path / "scr")},
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    bwrap, uv, path = result.stdout.splitlines()[-3:]
    assert (bwrap, uv) == (str(home_bin / "bwrap"), str(home_bin / "uv"))
    assert "/scr/" not in path


def test_every_env_script_is_checked():
    assert sorted(REPO_ROOT.glob("scripts/**/_env.sh")) == sorted(AGENT_JOB_ENV_SCRIPTS)
