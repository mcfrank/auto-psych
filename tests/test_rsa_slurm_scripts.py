"""The RSA run-1 Slurm scripts (scripts/rsa/slurm/): syntax, the agent tree's
isolation, the agents' `uv run` shim and the status report.

Sherlock is the only place these jobs run, so this checks what can be checked
off the cluster: every script parses, the exclude file withholds the data, the
docs, tests and scripts and the evaluation code, a tree built the way the array
builds it contains neither the recovery ground truth nor a held-out test set
(with rsync when it is installed; otherwise with a copy that applies the same
exclude rules), the agents' self-check imports from the tree through the shim,
and check_agent_tree.sh refuses each kind of leak.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from pyprojroot import here

REPO = here()
SLURM = REPO / "scripts" / "rsa" / "slurm"
EXCLUDE_FILE = SLURM / "agent_tree.exclude"
SR_EXCLUDE_FILE = REPO / "scripts" / "subjective_randomness" / "slurm" / "agent_tree.exclude"
SEEDS_REL = "src/pipelines/outer_loop/projects/rsa_reference/seed_models"
SEEDS = REPO / SEEDS_REL
REAL_DATA = REPO / "src/pipelines/outer_loop/projects/rsa_reference/data"

# Paths no RSA agent may see (PI decisions 2026-10-07 and the task brief).
FORBIDDEN = [
    "data",
    "data/rsa/external",
    "src/pipelines/outer_loop/projects/rsa_reference/data",
    "src/pipelines/outer_loop/projects/rsa_reference/data/pragmods_trials.csv",
    "src/pipelines/outer_loop/projects/rsa_reference/experiment",
    "src/rsa/ingest",
    "src/rsa/ingest/run.py",
    "src/rsa/split.py",
    "src/rsa/simulate.py",
    "src/rsa/recovery.py",
    "src/rsa/evaluate_heldout.py",
    "docs",
    "docs/auto_rsa/PLAN.md",
    "tests",
    "scripts",
    "scripts/rsa/slurm/agent_tree.exclude",
    "CLAUDE.md",
    "AGENTS.md",
    ".secrets",
    ".git",
]
# What the agents need: the self-check and what it imports, the seed models,
# the pyprojroot sentinel and opencode's config.
NEEDED = [
    "src/rsa/loop/check_candidate.py",
    "src/rsa/loop/gates.py",
    "src/rsa/fit.py",
    "src/rsa/model_file.py",
    "src/models/pymc_inference.py",
    "src/pipelines/inner_loop/model_zoo.py",
    f"{SEEDS_REL}/models_manifest.yaml",
    f"{SEEDS_REL}/rsa_l1.py",
    ".here",
    "opencode.json",
    "pyproject.toml",
]
RECOVERY_SALIENCE_SEEDS = ["rsa_l1_salience", "rsa_l1_shared_prior"]


# --- rsync's exclude semantics, for the patterns agent_tree.exclude uses ----------


def _rules(path: Path):
    rules = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rules.append((line.startswith("/"), line.endswith("/"), line.strip("/")))
    return rules


def _matches(rel: str, is_dir: bool, rules) -> bool:
    parts = rel.split("/")
    for anchored, dir_only, pattern in rules:
        if dir_only and not is_dir:
            continue
        pparts = pattern.split("/")
        if anchored or len(pparts) > 1:
            if len(pparts) == len(parts) and all(fnmatch.fnmatchcase(a, p) for a, p in zip(parts, pparts)):
                return True
        elif fnmatch.fnmatchcase(parts[-1], pattern):
            return True
    return False


def _excluded(rel: str, rules, is_dir: bool | None = None) -> bool:
    """True when rsync --exclude-from would leave ``rel`` out (itself or an ancestor)."""
    parts = rel.split("/")
    if any(_matches("/".join(parts[:i]), True, rules) for i in range(1, len(parts))):
        return True
    if is_dir is None:
        is_dir = (REPO / rel).is_dir()
    return _matches(rel, is_dir, rules)


def _tracked_files():
    out = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, capture_output=True, check=True).stdout
    return [p for p in out.decode().split("\0") if p and (REPO / p).is_file()]


def _copy_tree(dest: Path) -> Path:
    """The agent tree as rsync would build it from the checkout's tracked files."""
    rules = _rules(EXCLUDE_FILE)
    for rel in _tracked_files():
        if _excluded(rel, rules, is_dir=False):
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, target)
    return dest


def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def _scrub(tree: Path, seeds):
    return _run(["bash", str(SLURM / "scrub_agent_tree.sh"), str(tree), sys.executable, *seeds])


def _check(tree: Path, *args):
    return _run(["bash", str(SLURM / "check_agent_tree.sh"), str(tree), *map(str, args)])


@pytest.fixture
def held_out(tmp_path):
    """A stand-in for $DATA_ROOT: a test split and the simulation's provenance."""
    data = tmp_path / "data_root" / "recovery_salience"
    data.mkdir(parents=True)
    # Any real trials will do: what matters is that their bytes never reach a tree.
    shutil.copyfile(REAL_DATA / "sikos_2021_trials.csv", data / "test.csv")
    (data / "provenance.json").write_text(json.dumps({"ground_truth": str(SEEDS / "rsa_l1_salience.py")}))
    return data


@pytest.fixture(scope="module")
def scrubbed_tree(tmp_path_factory):
    tree = _copy_tree(tmp_path_factory.mktemp("agent_trees") / "0123abcd" / "repo")
    done = _scrub(tree, RECOVERY_SALIENCE_SEEDS)
    assert done.returncode == 0, done.stdout + done.stderr
    return tree


# --- the scripts --------------------------------------------------------------------


SCRIPTS = sorted(SLURM.glob("*.sh")) + sorted(SLURM.glob("*.sbatch"))


def test_the_sweep_has_its_scripts():
    names = {p.name for p in SCRIPTS}
    for name in ("_env.sh", "_cells.sh", "prepare_data.sh", "stage_code.sh", "rsa_setup.sbatch",
                 "rsa_loop_array.sbatch", "build_agent_tree.sh", "scrub_agent_tree.sh",
                 "check_agent_tree.sh", "rsa_status.sh", "submit.sh"):
        assert name in names
    assert EXCLUDE_FILE.is_file()


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_every_script_parses(script):
    result = _run(["bash", "-n", str(script)])
    assert result.returncode == 0, result.stderr


def test_the_array_runs_the_production_settings():
    text = (SLURM / "rsa_loop_array.sbatch").read_text()
    for setting in ('MAX_ITERATIONS="${MAX_ITERATIONS:-$ROUNDS}"', 'CANDIDATE_COUNT="${CANDIDATE_COUNT:-6}"',
                    'NUM_WARMUP="${NUM_WARMUP:-1000}"', 'NUM_SAMPLES="${NUM_SAMPLES:-1000}"',
                    'NUM_CHAINS="${NUM_CHAINS:-4}"', 'AGENT_MODEL="${AGENT_MODEL:-google/gemini-3.8-flash}"',
                    'AGENT_TIMEOUT_SEC="${AGENT_TIMEOUT_SEC:-2400}"', "--agent-root", "src.rsa.evaluate_heldout",
                    "src.rsa.recovery"):
        assert setting in text
    # Agents are sandboxed: the loop's default, never switched off here.
    assert "--no-sandbox" not in text


def test_resources_are_the_measured_sizes_and_overridable():
    text = (SLURM / "submit.sh").read_text()
    for knob in ('CPUS_PER_TASK="${CPUS_PER_TASK:-4}"', 'MEM="${MEM:-36G}"', 'TIME="${TIME:-24:00:00}"',
                 'SETUP_TIME="${SETUP_TIME:-01:00:00}"', 'PARTITION="${PARTITION:-mcfrank}"'):
        assert knob in text
    # --qos=long is not on the account (Sherlock run 1): never added.
    assert not re.search(r"sbatch[^\n]*--qos", text) and "qos_args" not in text


def test_a_long_limit_on_normal_stops_before_submitting(tmp_path):
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "sbatch").write_text("#!/bin/sh\necho SUBMITTED >&2; echo 1\n")
    (fake / "sbatch").chmod(0o755)
    env = dict(os.environ, PATH=f"{fake}:{os.environ['PATH']}", WORK_ROOT=str(tmp_path / "w"),
               PARTITION="normal", TIME="3-00:00:00", SKIP_SETUP="1")
    out = subprocess.run(["bash", str(SLURM / "submit.sh")], capture_output=True, text=True, env=env)
    assert out.returncode != 0 and "PARTITION=mcfrank" in out.stderr and "SUBMITTED" not in out.stderr


# --- the cell map ---------------------------------------------------------------------


def _cells():
    script = f'source "{SLURM / "_cells.sh"}"; for t in $(seq 0 $((RSA_N_CELLS - 1))); do ' \
             'rsa_cell $t; echo "$t|$CELL|$CONDITION|$REPLICATE|$GT|$EXCLUDED_SEEDS|$LOOP_SEED|$ROUNDS"; done'
    out = _run(["bash", "-c", script], check=True).stdout.split("\n")
    return [line.split("|") for line in out if line]


def test_run2_cells_are_three_real_replicates_and_two_of_each_recovery():
    """PI decision 2026-10-08: real x 3 x 8 rounds, recovery x 2 x 4 rounds;
    the long real cells come first in the array."""
    cells = _cells()
    assert [c[1] for c in cells] == ["real_rep1", "real_rep2", "real_rep3",
                                     "recovery_literal_rep1", "recovery_salience_rep1",
                                     "recovery_literal_rep2", "recovery_salience_rep2"]
    by_condition = {c[2]: c for c in cells}
    assert by_condition["real"][4:6] == ["", ""]
    assert by_condition["recovery_literal"][4:6] == ["literal_listener", "literal_listener"]
    assert by_condition["recovery_salience"][4:6] == ["rsa_l1_salience", "rsa_l1_salience rsa_l1_shared_prior"]
    assert {c[2]: c[7] for c in cells} == {"real": "8", "recovery_literal": "4", "recovery_salience": "4"}
    # Replicates differ in the loop seed, not in the data.
    assert sorted(c[6] for c in cells if c[2] == "real") == ["0", "1", "2"]
    for cond in ("recovery_literal", "recovery_salience"):
        assert sorted(c[6] for c in cells if c[2] == cond) == ["0", "1"]


def test_replicates_and_rounds_can_be_overridden():
    script = f'export REAL_REPLICATES=2 RECOVERY_REPLICATES=1 REAL_ROUNDS=5; source "{SLURM / "_cells.sh"}"; ' \
             'echo $RSA_N_CELLS; rsa_cell 0; echo $ROUNDS'
    out = _run(["bash", "-c", script], check=True).stdout.split()
    assert out == ["4", "5"]


def test_every_excluded_seed_is_a_seed_model():
    manifest = (SEEDS / "models_manifest.yaml").read_text()
    for cell in _cells():
        for seed in cell[5].split():
            assert (SEEDS / f"{seed}.py").is_file()
            assert f"name: {seed}\n" in manifest


@pytest.mark.parametrize("has_seed", [True, False])
def test_the_loop_seed_flag_is_detected(tmp_path, has_seed):
    fake = tmp_path / "python"
    flag = "│ --seed INT             (default: 0)  │" if has_seed else "│ --seed-models PATH  │"
    fake.write_text(f"#!/bin/bash\necho '│ --exclude-seeds [STR [STR ...]] │'\necho '{flag}'\n")
    fake.chmod(0o755)
    result = _run(["bash", "-c", f'source "{SLURM / "_cells.sh"}"; rsa_loop_cli_has_seed "{fake}"'])
    assert (result.returncode == 0) is has_seed


# --- the exclude file -----------------------------------------------------------------


@pytest.mark.parametrize("rel", FORBIDDEN)
def test_the_exclude_file_withholds(rel):
    assert _excluded(rel, _rules(EXCLUDE_FILE), is_dir=not Path(rel).suffix), rel


@pytest.mark.parametrize("rel", NEEDED)
def test_the_exclude_file_keeps_what_agents_need(rel):
    assert (REPO / rel).exists()
    assert not _excluded(rel, _rules(EXCLUDE_FILE)), rel


def test_no_tracked_csv_or_data_survives_the_exclude_file():
    rules = _rules(EXCLUDE_FILE)
    kept = [p for p in _tracked_files() if not _excluded(p, rules, is_dir=False)]
    assert not [p for p in kept if p.endswith(".csv") or "/data/" in f"/{p}"]


def test_every_subjective_randomness_rule_carries_over():
    """Each SR exclusion is in the RSA file, or inside a directory it excludes."""
    rsa_lines = {line for _, _, line in _rules(EXCLUDE_FILE)}
    rsa_rules = _rules(EXCLUDE_FILE)
    for anchored, _, line in _rules(SR_EXCLUDE_FILE):
        covered = line in rsa_lines or (anchored and _excluded(line.replace("*", "x"), rsa_rules, is_dir=True))
        assert covered, f"{line} (from the subjective-randomness exclude file) is not excluded"


# --- a tree built as the array builds it -----------------------------------------------


def test_the_scrubbed_tree_has_no_ground_truth_and_passes_the_check(scrubbed_tree, held_out):
    tree = scrubbed_tree
    for seed in RECOVERY_SALIENCE_SEEDS:
        assert not (tree / SEEDS_REL / f"{seed}.py").exists()
    manifest = (tree / SEEDS_REL / "models_manifest.yaml").read_text()
    assert "rsa_l1_salience" not in manifest and "rsa_l1_shared_prior" not in manifest
    for kept in ("literal_listener", "rsa_l1", "rsa_l2"):
        assert (tree / SEEDS_REL / f"{kept}.py").is_file() and f"name: {kept}\n" in manifest
    for rel in FORBIDDEN:
        assert not (tree / rel).exists(), rel
    assert not list(tree.rglob("*.csv"))
    gt_args = [a for s in RECOVERY_SALIENCE_SEEDS for a in ("--gt-file", SEEDS / f"{s}.py")]
    result = _check(tree, *gt_args, "--forbid-file", held_out / "test.csv",
                    "--forbid-file", held_out / "provenance.json")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "agent tree OK" in result.stdout


def test_a_real_cell_withholds_no_seed_and_passes_the_check(tmp_path, held_out):
    # A real cell passes no --gt-file. el7's bash 4.2 treats an empty array as
    # unbound under `set -u`, so a bare "${gt_files[@]}" killed every real cell
    # of Sherlock run 1 (array 46898755); bash 5 here does not, hence the
    # source check as well as the run.
    source = (SLURM / "check_agent_tree.sh").read_text()
    for name in ("gt_files", "forbid_files"):
        assert f'in "${{{name}[@]}}"' not in source, name
    tree = _copy_tree(tmp_path / "agent_trees" / "4567cdef" / "repo")
    done = _scrub(tree, [])
    assert done.returncode == 0, done.stdout + done.stderr
    result = _check(tree, "--forbid-file", held_out / "test.csv")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "agent tree OK" in result.stdout


def test_the_self_check_imports_from_the_tree_through_the_uv_shim(scrubbed_tree):
    tree = scrubbed_tree
    env = dict(os.environ, PATH=f"{tree / '.agent_bin'}:{os.environ['PATH']}")
    code = "import src, src.rsa.loop.check_candidate as m; print(m.__file__)"
    result = _run(["uv", "run", "python", "-c", code], cwd=tree, env=env)
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()).resolve().is_relative_to(tree.resolve())
    # The documented command parses (its --help needs no data).
    result = _run(["uv", "run", "python", "-m", "src.rsa.loop.check_candidate", "--help"], cwd=tree, env=env)
    assert result.returncode == 0 and "--responses" in result.stdout, result.stderr


@pytest.mark.parametrize("command", [["pip", "install", "numpy"], ["sync"], ["add", "x"], ["run", "--with", "x", "python"],
                                     ["run", "pytest"]])
def test_the_uv_shim_refuses_everything_but_running_python(scrubbed_tree, command):
    result = _run([str(scrubbed_tree / ".agent_bin" / "uv"), *command], cwd=scrubbed_tree)
    assert result.returncode == 2
    assert "not available here" in result.stderr


def test_the_uv_shim_accepts_uv_run_options(scrubbed_tree):
    result = _run([str(scrubbed_tree / ".agent_bin" / "uv"), "run", "--frozen", "python", "-c", "print(41 + 1)"],
                  cwd=scrubbed_tree)
    assert result.returncode == 0 and result.stdout.strip() == "42", result.stderr


def _leaky_copy(src: Path, dest: Path) -> Path:
    shutil.copytree(src, dest, symlinks=True)
    return dest


@pytest.mark.parametrize("leak", ["renamed_copy", "named_file", "manifest_entry", "test_in_results",
                                  "test_renamed", "provenance", "data_dir", "csv", "snapshots_on", "split_module"])
def test_the_check_refuses_each_leak(scrubbed_tree, held_out, tmp_path, leak):
    tree = _leaky_copy(scrubbed_tree, tmp_path / "tree")
    gt = SEEDS / "rsa_l1_salience.py"
    if leak == "renamed_copy":
        shutil.copyfile(gt, tree / "src" / "rsa" / "helper.py")
    elif leak == "named_file":
        (tree / "src" / "rsa_l1_salience.py").write_text("# a rewrite\n")
    elif leak == "manifest_entry":
        with (tree / SEEDS_REL / "models_manifest.yaml").open("a") as f:
            f.write("  - name: rsa_l1_salience\n    rationale: x\n")
    elif leak == "test_in_results":
        (tree / "_runs" / "loop").mkdir(parents=True)
        shutil.copyfile(held_out / "test.csv", tree / "_runs" / "loop" / "test.csv")
    elif leak == "test_renamed":
        (tree / "_runs" / "loop").mkdir(parents=True)
        shutil.copyfile(held_out / "test.csv", tree / "_runs" / "loop" / "heldout_rows.dat")
    elif leak == "provenance":
        shutil.copyfile(held_out / "provenance.json", tree / "src" / "notes.json")
    elif leak == "data_dir":
        (tree / "src/pipelines/outer_loop/projects/rsa_reference/data").mkdir()
    elif leak == "csv":
        (tree / "src" / "rows.csv").write_text("a,b\n1,2\n")
    elif leak == "snapshots_on":
        (tree / "opencode.json").write_text(json.dumps({"snapshot": True}))
    elif leak == "split_module":
        (tree / "src" / "rsa" / "split.py").write_text("# the split\n")
    result = _check(tree, "--gt-file", gt, "--forbid-file", held_out / "test.csv",
                    "--forbid-file", held_out / "provenance.json")
    assert result.returncode == 1, result.stdout
    assert "ERROR" in result.stderr


def test_the_scrub_refuses_a_seed_the_manifest_does_not_list(tmp_path):
    tree = tmp_path / "tree"
    shutil.copytree(REPO / "src", tree / "src", ignore=shutil.ignore_patterns("__pycache__", "data"))
    result = _scrub(tree, ["no_such_seed"])
    assert result.returncode != 0
    assert "no_such_seed" in result.stderr and "ModuleNotFoundError" not in result.stderr


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync is not installed (it is on Sherlock)")
def test_the_rsync_build_withholds_the_ground_truth_and_the_test_set(tmp_path, held_out):
    tree = tmp_path / "agent_trees" / "89abcdef" / "repo"
    result = _run(["bash", str(SLURM / "build_agent_tree.sh"), str(REPO), str(tree), sys.executable,
                   *RECOVERY_SALIENCE_SEEDS])
    assert result.returncode == 0, result.stdout + result.stderr
    for seed in RECOVERY_SALIENCE_SEEDS:
        assert not list(tree.rglob(f"{seed}.py"))
    assert not list(tree.rglob("test.csv")) and not list(tree.rglob("*.csv"))
    for rel in FORBIDDEN:
        assert not (tree / rel).exists(), rel
    gt_args = [a for s in RECOVERY_SALIENCE_SEEDS for a in ("--gt-file", SEEDS / f"{s}.py")]
    result = _check(tree, *gt_args, "--forbid-file", held_out / "test.csv",
                    "--forbid-file", held_out / "provenance.json")
    assert result.returncode == 0, result.stdout + result.stderr
    # A rebuild keeps the loop's results (the protect filter) and stays clean.
    (tree / "_runs" / "loop" / ".fit_cache").mkdir(parents=True)
    (tree / "_runs" / "loop" / ".fit_cache" / "m.nc").write_text("fit")
    result = _run(["bash", str(SLURM / "build_agent_tree.sh"), str(REPO), str(tree), sys.executable,
                   *RECOVERY_SALIENCE_SEEDS])
    assert result.returncode == 0, result.stderr
    assert (tree / "_runs" / "loop" / ".fit_cache" / "m.nc").exists()


# --- the status report --------------------------------------------------------------------


def test_the_status_report_summarises_each_cell(tmp_path):
    root = tmp_path / "rsa_run1"
    results = root / "agent_trees" / "aa" / "repo" / "_runs" / "loop"
    results.mkdir(parents=True)
    cell = root / "cells" / "recovery_salience_rep1"
    cell.mkdir(parents=True)
    (cell / "results").symlink_to(results)
    (cell / "cell.json").write_text(json.dumps({"max_iterations": 5}))
    (cell / "last_exit").write_text("0 job=7 at=now\n")
    (cell / "DONE").write_text("now\n")
    (results / "history.json").write_text(json.dumps(
        [{"round": -1, "best_model": "rsa_l1"}] + [{"round": r, "best_model": "m_new"} for r in range(6)]))
    (results / "export.json").write_text(json.dumps({"best_model": "m_new", "live": ["m_new"]}))
    ledger = [{"outcome": o} for o in ["admitted", "admitted", "rejected", "pruned"]]
    (results / "attempted_hypotheses.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ledger))
    usage = [{"input_tokens": 1_000_000, "output_tokens": 500_000, "cost_usd": 1.25},
             {"usage_missing": True}]
    (results / "token_usage.jsonl").write_text("".join(json.dumps(u) + "\n" for u in usage))
    (results / "recovery").mkdir()
    (results / "recovery" / "recovery.json").write_text(json.dumps({"recovered": True}))
    failed = root / "cells" / "real_rep1"
    failed.mkdir(parents=True)
    (failed / "last_exit").write_text("1 job=8 at=then\n")
    env = dict(os.environ, PATH=f"{tmp_path / 'nobin'}:{os.environ['PATH']}")
    result = _run(["bash", str(SLURM / "rsa_status.sh"), str(root)], env=env)
    assert result.returncode == 0, result.stderr
    lines = {line.split()[1]: line for line in result.stdout.splitlines() if line[:1].isdigit()}
    assert len(lines) == 7
    done = lines["recovery_salience_rep1"].split()
    assert done[2] == "done" and "m_new" in done and "2/1/1" in done and "1.5M" in done
    assert "$1.25+" in done and done[-1] == "yes"
    assert lines["real_rep1"].split()[2] == "failed"
    assert lines["real_rep2"].split()[2] == "pending"


def test_the_sweep_root_default_lives_in_one_place():
    # After run 1, submit.sh, rsa_status.sh and rsa_loop_array.sbatch still said
    # rsa_run1 while _env.sh said rsa_run2; run 2's first submit went to run 1's root.
    named = [p.name for p in SCRIPTS if re.search(r"/auto-psych/rsa_run\d|rsa_run\d+\}", p.read_text())]
    assert named == []
    assert 'RSA_SWEEP_NAME="rsa_run' in (SLURM / "_cells.sh").read_text()


def test_jax_runs_on_one_cpu_per_process_and_agents_inherit_it():
    # Research Computing (2026-10-09): XLA sized its thread pool by the node, not the job.
    from src.rsa.loop.fitting import SINGLE_THREAD_XLA_FLAGS
    from src.runtime.agent_sandbox import AGENT_ENV_NAMES, agent_environment

    assert SINGLE_THREAD_XLA_FLAGS in (SLURM / "_env.sh").read_text()
    assert "XLA_FLAGS" in AGENT_ENV_NAMES
    assert agent_environment({"XLA_FLAGS": SINGLE_THREAD_XLA_FLAGS}, "opencode")["XLA_FLAGS"] == SINGLE_THREAD_XLA_FLAGS


def test_jax_compiles_are_cached_outside_every_agent_tree():
    text = (SLURM / "_env.sh").read_text()
    assert 'JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$(dirname "$WORK_ROOT")/jax_cache}"' in text
    from src.runtime.agent_sandbox import AGENT_ENV_NAMES

    assert "JAX_COMPILATION_CACHE_DIR" not in AGENT_ENV_NAMES


@pytest.mark.parametrize("name", ["chains.sbatch", "design.sbatch"])
def test_single_process_jobs_ask_for_one_cpu_and_memory_that_buys_no_more(name):
    # mcfrank's MaxMemPerCPU is 8000 MB: --mem=32GB with -c 4 was allocated 5 CPUs.
    text = (SLURM / name).read_text()
    assert "#SBATCH --cpus-per-task=1\n" in text and "#SBATCH --mem=7GB\n" in text


def test_openblas_sizes_its_pool_to_one_thread():
    assert "export OPENBLAS_NUM_THREADS=1" in (SLURM / "_env.sh").read_text()
