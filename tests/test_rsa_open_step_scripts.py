"""The step jobs' shell side (scripts/rsa/slurm/_step_mode.sh): the continuation script and what
a step does with run_chain's exit code, against a stand-in sbatch."""

import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "rsa" / "slurm" / "_step_mode.sh"


def _setup(tmp_path, *extra):
    cell = tmp_path / "cells" / "open_chain_0"
    cell.mkdir(parents=True)
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "sbatch").write_text(f'#!/bin/bash\nprintf "%s\\n" "$@" > {tmp_path}/sbatch_args\necho "777;sherlock"\n')
    (fake / "sbatch").chmod(0o755)
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "USER": "u", "CELL": "open_chain_0", "CELL_DIR": str(cell),
           "WORK_ROOT": str(tmp_path / "work"), "REPO": "/r", "SLURM_DIR": "/h/scripts/rsa/slurm", "VENV_PY": "/v/py",
           "HARNESS_REPO": "/h", "JAX_COMPILATION_CACHE_DIR": "/j", "SLURM_JOB_NAME": "rsa_open_open_c0",
           "SLURM_JOB_ID": "41", "STEP_MODE": "1", "ARM": "open", "CHAIN": "0", "CONFIG": "/cfg dir/live.yaml"}
    cmd = (f'source {SCRIPT}; rsa_step_mode /r/scripts/rsa/slurm/open_live.sbatch ARM CHAIN CONFIG UNSET_ONE; '
           'printf "%s\\n" "${STEP_ENV[@]}"; ' + " ".join(extra))
    out = subprocess.run(["bash", "-c", cmd], env=env, capture_output=True, text=True)
    cont = cell / "continuation.sh"
    if cont.exists():  # run it against the stand-in (no login shell here)
        text = cont.read_text().replace("env -i ", f"env PATH={fake}:/usr/bin:/bin ").replace("bash -lc", "bash -c")
        cont.write_text(text)
    return out, cell, env


def test_the_continuation_submits_the_same_script_after_the_dependency_with_its_settings(tmp_path):
    out, cell, env = _setup(tmp_path)
    assert out.returncode == 0, out.stderr
    step_env = dict(line.split("=", 1) for line in out.stdout.splitlines())
    assert step_env["RSA_OPEN_STEP_MODE"] == "1" and step_env["RSA_OPEN_FIT_BACKEND"] == "slurm"
    assert step_env["RSA_OPEN_CHAIN_JOB"] == "rsa_open_open_c0" and step_env["RSA_OPEN_FIT_PARTITION"] == "mcfrank,owners"
    assert step_env["RSA_OPEN_FIT_SCRIPT"] == "/h/scripts/rsa/slurm/fit_array.sbatch"
    jid = subprocess.run(["bash", str(cell / "continuation.sh"), "afterany:5:6,singleton"], capture_output=True,
                         text=True, check=True).stdout.strip()
    assert jid == "777"
    args = (tmp_path / "sbatch_args").read_text().splitlines()
    assert args[-1] == "/r/scripts/rsa/slurm/open_live.sbatch" and "--dependency=afterany:5:6,singleton" in args
    assert "--no-requeue" in args and "--partition=mcfrank" in args and "--job-name=rsa_open_open_c0" in args
    export = next(a for a in args if a.startswith("--export="))
    assert export == f"--export=ALL,STEP_SETTINGS={cell}/step_settings.sh"
    got = subprocess.run(["bash", "-c", f"source {cell}/step_settings.sh; env"], env={"PATH": os.environ["PATH"]},
                         capture_output=True, text=True, check=True).stdout.splitlines()
    for want in ("CONFIG=/cfg dir/live.yaml", "STEP_MODE=1", "STEP_FIT_PARTITION=mcfrank,owners", "ARM=open", "CHAIN=0",
                 f"WORK_ROOT={tmp_path / 'work'}"):
        assert want in got
    assert not any(line.startswith("UNSET_ONE=") for line in got)


def test_a_cell_path_with_a_comma_is_refused(tmp_path):
    cell = tmp_path / "c,d"
    cell.mkdir()
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "CELL": "x", "CELL_DIR": str(cell), "WORK_ROOT": str(tmp_path),
           "REPO": "/r", "SLURM_DIR": "/h", "VENV_PY": "/v", "HARNESS_REPO": "/h", "JAX_COMPILATION_CACHE_DIR": "/j",
           "SLURM_JOB_NAME": "n", "ARM": "a,b"}  # a comma in a setting is fine
    out = subprocess.run(["bash", "-c", f"source {SCRIPT}; rsa_step_mode /s ARM"], env=env, capture_output=True, text=True)
    assert out.returncode != 0 and "comma" in out.stderr


def test_exit_codes_yield_refuse_and_pause(tmp_path):
    for rc, want in ((6, 0), (7, 7), (3, 3), (8, 15)):
        out, cell, _ = _setup(tmp_path / str(rc), f"rsa_step_exit {rc}")
        assert out.returncode == want, (rc, out.stdout, out.stderr)
    out, cell, env = _setup(tmp_path / "5")
    fake = tmp_path / "5" / "bin"
    run = subprocess.run(["bash", "-c", f"CONT={cell}/continuation.sh; source {SCRIPT}; rsa_step_exit 5"],
                         env={**env, "PATH": f"{fake}:{os.environ['PATH']}"}, capture_output=True, text=True)
    assert run.returncode == 0 and "continues in job 777" in run.stdout, run.stderr
    assert "--dependency=singleton" in (tmp_path / "5" / "sbatch_args").read_text()


def test_a_finished_experiment_stops_the_chain_for_a_person_and_mails(tmp_path):
    out, cell, _ = _setup(tmp_path, "rsa_step_exit 8")
    assert out.returncode == 15, out.stderr
    assert "launch the next: CLEAR_STOP=1 STEP_MODE=1" in (cell / "STOP").read_text()


def test_a_person_s_launch_refuses_while_another_step_of_the_chain_is_queued(tmp_path):
    fake = tmp_path / "squeue_bin"
    fake.mkdir()
    (fake / "squeue").write_text('#!/bin/bash\nprintf "41\\n57\\n"\n')  # itself (41) and a queued step (57)
    (fake / "squeue").chmod(0o755)
    cell = tmp_path / "cell"
    cell.mkdir()
    env = {"PATH": f"{fake}:{os.environ['PATH']}", "HOME": str(tmp_path), "CELL": "open_chain_0", "CELL_DIR": str(cell),
           "WORK_ROOT": str(tmp_path), "REPO": "/r", "SLURM_DIR": "/h", "VENV_PY": "/v", "HARNESS_REPO": "/h",
           "JAX_COMPILATION_CACHE_DIR": "/j", "SLURM_JOB_NAME": "rsa_open_open_c0", "SLURM_JOB_ID": "41",
           "RSA_STEP_NEW_LAUNCH": "1"}
    out = subprocess.run(["bash", "-c", f"source {SCRIPT}; rsa_step_mode /s"], env=env, capture_output=True, text=True)
    assert out.returncode == 1 and "already has a step queued or running (jobs: 57)" in out.stderr
    (fake / "squeue").write_text('#!/bin/bash\nprintf "41\\n"\n')  # only itself
    out = subprocess.run(["bash", "-c", f"source {SCRIPT}; rsa_step_mode /s; printf '%s\\n' \"${{STEP_ENV[@]}}\""],
                         env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "RSA_OPEN_NEW_LAUNCH=1" in out.stdout.splitlines()
