#!/usr/bin/env python3
"""Submit auto-psych pipeline jobs to a SLURM cluster from a YAML manifest.

Reads remote_jobs/.env.local for REMOTE_HOST / REMOTE_PROJECT_DIR / REMOTE_SCRATCH_DIR,
expands the manifest's matrix cross-product, ssh+git-pulls the cluster repo, and submits
one sbatch per cell. Records the submission in remote_jobs/experiments.md (committed) and
remote_jobs/.runs/<timestamp>_<job_id>.json (gitignored) for the syncer to consult.

Fire-and-forget: returns as soon as all sbatch calls have been issued. Use
remote_jobs/sync_from_remote.sh later to pull batches back.

Usage:
  remote_jobs/submit.py remote_jobs/jobs/foo.yaml
  remote_jobs/submit.py remote_jobs/jobs/foo.yaml --allow-dirty
  remote_jobs/submit.py remote_jobs/jobs/foo.yaml --dry-run

Use --dry-run to print the planned sbatch invocations and the experiments.md entry without
contacting the cluster (useful for verifying manifest expansion in tests).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
REMOTE_DIR = REPO_ROOT / "remote_jobs"
EXPERIMENTS_MD = REMOTE_DIR / "experiments.md"
RUNS_DIR = REMOTE_DIR / ".runs"
ENTRIES_OPEN = "<!-- entries:start -->"
ENTRIES_CLOSE = "<!-- entries:end -->"


# -------------------- env loading -----------------------------------------


def _load_env_local() -> Dict[str, str]:
    """Parse remote_jobs/.env.local KEY=VAL lines (quotes stripped). Returns {} if absent."""
    env_path = REMOTE_DIR / ".env.local"
    out: Dict[str, str] = {}
    if not env_path.exists():
        return out
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        # Strip a single pair of matching surrounding quotes (single or double).
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        out[key] = val
    return out


def _resolve_env() -> Tuple[str, str, str]:
    """Pick REMOTE_HOST / REMOTE_PROJECT_DIR / REMOTE_SCRATCH_DIR from .env.local + os.environ."""
    env_file = _load_env_local()
    host = os.environ.get("REMOTE_HOST") or env_file.get("REMOTE_HOST", "sherlock")
    proj = os.environ.get("REMOTE_PROJECT_DIR") or env_file.get("REMOTE_PROJECT_DIR", "$HOME/auto-psych")
    scr = os.environ.get("REMOTE_SCRATCH_DIR") or env_file.get("REMOTE_SCRATCH_DIR", "$SCRATCH/auto-psych")
    return host, proj, scr


# -------------------- git state -------------------------------------------


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip()


def _git_state() -> Dict[str, Any]:
    """Capture commit, dirty flag, branch, upstream sync state for the experiments.md entry.

    We only care about tracked-but-modified files for the dirty check: those are the changes
    that won't be visible on the cluster after `git pull`. Untracked files are excluded with
    --untracked-files=no since they aren't part of any commit and so cannot drift remote-side.

    We also exempt files this submitter writes itself (`remote_jobs/experiments.md`), since
    appending to that journal would otherwise immediately make the next submission appear
    dirty (a chicken-and-egg problem). The user is expected to commit the journal periodically.
    """
    head_full = _git("rev-parse", "HEAD") or ""
    head_short = head_full[:7] if head_full else "nogit"
    porcelain = _git("status", "--porcelain", "--untracked-files=no")
    # Filter out submitter-managed files from the dirty signal.
    submitter_managed = {"remote_jobs/experiments.md"}
    significant_lines = []
    for line in porcelain.splitlines():
        # Porcelain format: "XY filename" (XY is two-char status code, then space, then path).
        if len(line) <= 3:
            continue
        path = line[3:].strip()
        if path in submitter_managed:
            continue
        significant_lines.append(line)
    dirty = bool(significant_lines)
    porcelain = "\n".join(significant_lines)
    branch = _git("rev-parse", "--abbrev-ref", "HEAD") or "DETACHED"
    upstream = _git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}") or ""
    upstream_ahead = ""
    if upstream:
        ahead_behind = _git("rev-list", "--left-right", "--count", f"{upstream}...HEAD")
        if ahead_behind:
            try:
                behind, ahead = ahead_behind.split()
                upstream_ahead = f"ahead={ahead} behind={behind}"
            except ValueError:
                upstream_ahead = ahead_behind
    return {
        "commit": head_full,
        "short": head_short,
        "dirty": dirty,
        "branch": branch,
        "upstream": upstream,
        "upstream_status": upstream_ahead,
        "porcelain": porcelain,
    }


def _git_is_pushed(state: Dict[str, Any]) -> bool:
    """True iff HEAD has an upstream and is not ahead of it."""
    if not state["upstream"]:
        return False
    if "ahead=0" in state["upstream_status"]:
        return True
    # If upstream tracking can't tell us, fall back to "not ahead".
    return state["upstream_status"] in ("", "behind=0 ahead=0")


# -------------------- manifest --------------------------------------------


# Matches valid SLURM partition / job names: letters, digits, underscore, hyphen, dot.
_JOB_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_TOP_LEVEL_KEYS = {"name", "project", "mode", "n_participants", "max_retries", "matrix", "slurm"}
# Pipeline scalar args we know how to forward (top-level only; not in matrix).
_PIPELINE_TOP_KEYS = {"project", "mode", "n_participants", "max_retries"}
_VALID_MODES = {"simulated_participants", "simulated_participants_nobrowser", "live", "test_prolific"}


def _load_manifest(path: Path) -> Dict[str, Any]:
    import yaml

    if not path.exists():
        sys.exit(f"submit.py: manifest not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        sys.exit(f"submit.py: manifest must be a YAML mapping: {path}")
    return data


def _validate_manifest(m: Dict[str, Any], path: Path) -> None:
    extra = set(m.keys()) - _TOP_LEVEL_KEYS
    if extra:
        sys.exit(f"submit.py: unknown top-level keys in {path}: {sorted(extra)}")
    for required in ("name", "project"):
        if not m.get(required):
            sys.exit(f"submit.py: manifest is missing required field '{required}': {path}")
    if not _JOB_NAME_RE.match(str(m["name"])):
        sys.exit(f"submit.py: 'name' must match {_JOB_NAME_RE.pattern}: {m['name']!r}")
    mode = m.get("mode") or "simulated_participants"
    if mode not in _VALID_MODES:
        sys.exit(f"submit.py: invalid mode {mode!r} (must be one of {sorted(_VALID_MODES)})")
    matrix = m.get("matrix") or {}
    if matrix and not isinstance(matrix, dict):
        sys.exit("submit.py: 'matrix' must be a mapping of dim -> list")
    for k, v in matrix.items():
        if not isinstance(v, list) or len(v) == 0:
            sys.exit(f"submit.py: matrix dim {k!r} must be a non-empty list")
        if k in _PIPELINE_TOP_KEYS:
            sys.exit(f"submit.py: matrix dim {k!r} conflicts with a top-level field")
    slurm = m.get("slurm") or {}
    if slurm and not isinstance(slurm, dict):
        sys.exit("submit.py: 'slurm' must be a mapping")


def _expand_cells(matrix: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Cartesian product of matrix dims, preserving YAML key order. Returns [{}] when empty."""
    if not matrix:
        return [{}]
    keys = list(matrix.keys())
    values = [matrix[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


# -------------------- per-cell command building ---------------------------


def _pipeline_args_for_cell(manifest: Dict[str, Any], cell: Dict[str, Any]) -> List[str]:
    """Translate manifest top-level + cell vars into run_pipeline.py CLI args."""
    args: List[str] = []
    for key in ("project", "mode", "n_participants", "max_retries"):
        if key in manifest and manifest[key] is not None:
            args.extend([f"--{key.replace('_', '-')}", str(manifest[key])])
    for key, val in cell.items():
        if val is None:
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(val, bool):
            if val:
                args.append(flag)
            continue
        args.extend([flag, str(val)])
    return args


def _cell_label(cell: Dict[str, Any]) -> str:
    """Short, filesystem-safe label appended to job-name and log paths."""
    if not cell:
        return "single"
    parts = []
    for k, v in cell.items():
        v_str = "none" if v is None else str(v)
        v_str = re.sub(r"[^A-Za-z0-9._-]+", "_", v_str)
        parts.append(f"{k}-{v_str}")
    return "_".join(parts)


def _slurm_flags(slurm_cfg: Dict[str, Any], job_name: str, log_dir_remote: str) -> List[str]:
    """Build the sbatch resource flags (--time / --cpus-per-task / --mem / --partition / --output / --error / --job-name)."""
    out_path = f"{log_dir_remote}/{job_name}-%j.out"
    err_path = f"{log_dir_remote}/{job_name}-%j.err"
    flags: List[str] = [
        f"--job-name={job_name}",
        f"--output={out_path}",
        f"--error={err_path}",
    ]
    key_map = {
        "time": "--time",
        "cpus_per_task": "--cpus-per-task",
        "mem": "--mem",
        "partition": "--partition",
        "account": "--account",
        "qos": "--qos",
        "nodes": "--nodes",
        "ntasks": "--ntasks",
    }
    for yaml_key, sbatch_flag in key_map.items():
        if yaml_key in slurm_cfg and slurm_cfg[yaml_key] is not None:
            flags.append(f"{sbatch_flag}={slurm_cfg[yaml_key]}")
    return flags


def _build_sbatch_remote_command(
    *,
    sbatch_flags: List[str],
    pipeline_args: List[str],
    remote_project_dir: str,
    remote_scratch_dir: str,
) -> str:
    """Produce the shell command we send over ssh: cd repo, export env, sbatch --parsable.

    --parsable makes sbatch print just the job id (or 'jobid;cluster') so we can capture it.

    Note on quoting: sbatch_flags often contain `$SCRATCH/...` (in --output/--error). We must
    NOT shell-quote them or the remote bash will pass `$SCRATCH` literally instead of expanding
    it. The flags are constructed from a strictly-validated job name + a path we control, so
    leaving them unquoted is safe. The pipeline args (user-supplied via the YAML manifest) are
    shell-quoted to defend against unusual values.
    """
    quoted_pipeline = " ".join(shlex.quote(a) for a in pipeline_args)
    raw_flags = " ".join(sbatch_flags)
    # remote_project_dir / remote_scratch_dir come from .env.local as e.g. '$HOME/auto-psych';
    # leave them unquoted so bash expands $HOME / $SCRATCH on the cluster side.
    return (
        f"cd {remote_project_dir} && "
        f"export REMOTE_PROJECT_DIR={remote_project_dir} REMOTE_SCRATCH_DIR={remote_scratch_dir} && "
        f"sbatch --parsable {raw_flags} remote_jobs/pipeline.slurm {quoted_pipeline}"
    )


# -------------------- ssh helpers -----------------------------------------


def _ssh(host: str, command: str, *, dry_run: bool = False) -> Tuple[int, str, str]:
    """Run `ssh host command` and return (returncode, stdout, stderr). dry_run echoes only."""
    if dry_run:
        print(f"[dry-run] ssh {host} {command}")
        return 0, "DRYRUN-JOBID-0", ""
    result = subprocess.run(
        ["ssh", host, command], capture_output=True, text=True, check=False
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def _git_pull_remote(host: str, remote_project_dir: str, *, dry_run: bool) -> None:
    cmd = f"cd {remote_project_dir} && git fetch --quiet && git checkout --quiet . && git pull --ff-only"
    code, out, err = _ssh(host, cmd, dry_run=dry_run)
    if code != 0:
        sys.exit(f"submit.py: remote git pull failed (code {code})\nstdout:\n{out}\nstderr:\n{err}")
    if out:
        print(f"[remote git pull] {out}")
    if err:
        print(f"[remote git pull stderr] {err}")


# -------------------- experiments.md + .runs/ -----------------------------


def _format_experiment_entry(
    *,
    manifest: Dict[str, Any],
    manifest_path: Path,
    git_state: Dict[str, Any],
    host: str,
    remote_project_dir: str,
    remote_scratch_dir: str,
    submitted_cells: List[Dict[str, Any]],
) -> str:
    """Markdown for one submission. submitted_cells contains {label, args, job_id, log_out, log_err}."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    push_state = "pushed" if git_state["upstream"] and "ahead=0" in git_state["upstream_status"] else "NOT pushed"
    dirty_label = "dirty" if git_state["dirty"] else "clean"
    lines: List[str] = [
        f"## {ts} -- {manifest['name']}",
        f"- **Manifest**: `{manifest_path.relative_to(REPO_ROOT)}`",
        f"- **Local commit**: `{git_state['short']}` ({dirty_label}) on `{git_state['branch']}` ({push_state}, upstream `{git_state['upstream'] or 'none'}`)",
        f"- **Remote host**: `{host}` * **Project dir**: `{remote_project_dir}` * **Scratch dir**: `{remote_scratch_dir}`",
        f"- **Slurm jobs submitted** ({len(submitted_cells)}):",
    ]
    for cell in submitted_cells:
        args_str = " ".join(cell["args"]) if cell.get("args") else "(no extra args)"
        lines.append(f"  - `{cell['job_id']}` -- `{args_str}` -- log: `{cell['log_out']}`")
    lines.append(f"- **Pull when done**: `./remote_jobs/sync_from_remote.sh --project {manifest['project']}`")
    lines.append("")
    return "\n".join(lines)


def _append_experiments_entry(entry_md: str) -> None:
    """Insert a new entry between the entries:start and entries:end markers (newest on top)."""
    EXPERIMENTS_MD.parent.mkdir(parents=True, exist_ok=True)
    if not EXPERIMENTS_MD.exists():
        # Reconstruct skeleton if the journal got deleted (shouldn't happen normally).
        skeleton = (
            "# Cluster experiments log\n\n"
            "Submissions, newest on top.\n\n"
            f"{ENTRIES_OPEN}\n\n{ENTRIES_CLOSE}\n"
        )
        EXPERIMENTS_MD.write_text(skeleton, encoding="utf-8")
    text = EXPERIMENTS_MD.read_text(encoding="utf-8")
    if ENTRIES_OPEN not in text or ENTRIES_CLOSE not in text:
        # If the markers were lost, append at end of file as a safe fallback.
        EXPERIMENTS_MD.write_text(text.rstrip() + "\n\n" + entry_md, encoding="utf-8")
        return
    idx = text.index(ENTRIES_OPEN) + len(ENTRIES_OPEN)
    new_text = text[:idx] + "\n\n" + entry_md + text[idx:]
    EXPERIMENTS_MD.write_text(new_text, encoding="utf-8")


def _write_run_record(record: Dict[str, Any]) -> Path:
    """Persist the full submission record under remote_jobs/.runs/."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = record.get("manifest", {}).get("name") or "submission"
    job_ids = "_".join(c.get("job_id", "?") for c in record.get("cells", []))[:64] or "noid"
    path = RUNS_DIR / f"{ts}_{name}_{job_ids}.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return path


# -------------------- main ------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Submit auto-psych pipeline jobs from a YAML manifest.")
    parser.add_argument("manifest", type=Path, help="Path to the YAML manifest")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Submit even if the local working tree has uncommitted changes or is unpushed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned sbatch commands and the experiments.md entry without contacting the cluster.",
    )
    parser.add_argument(
        "--no-pull",
        action="store_true",
        help="Skip the remote `git pull` step (use when iterating with --allow-dirty and rsync separately).",
    )
    args = parser.parse_args(argv)

    manifest_path = args.manifest if args.manifest.is_absolute() else (REPO_ROOT / args.manifest).resolve()
    manifest = _load_manifest(manifest_path)
    _validate_manifest(manifest, manifest_path)

    git_state = _git_state()
    if not args.allow_dirty:
        if git_state["dirty"]:
            sys.exit(
                "submit.py: refusing to submit -- working tree is dirty. "
                "Commit or stash changes, or pass --allow-dirty.\n"
                f"Modified files:\n{git_state['porcelain']}"
            )
        if not _git_is_pushed(git_state):
            sys.exit(
                "submit.py: refusing to submit -- HEAD is not pushed to its upstream. "
                "Push the branch (or pass --allow-dirty) so the cluster's git pull sees this commit."
            )

    host, remote_project_dir, remote_scratch_dir = _resolve_env()
    log_dir_remote = f"{remote_scratch_dir}/logs"

    if not args.no_pull:
        _git_pull_remote(host, remote_project_dir, dry_run=args.dry_run)

    cells = _expand_cells(manifest.get("matrix") or {})
    slurm_cfg = manifest.get("slurm") or {}
    submitted: List[Dict[str, Any]] = []
    failures: List[str] = []

    for cell in cells:
        label = _cell_label(cell)
        job_name = f"{manifest['name']}__{label}"[:64] if label != "single" else manifest["name"]
        sbatch_flags = _slurm_flags(slurm_cfg, job_name=job_name, log_dir_remote=log_dir_remote)
        pipeline_args = _pipeline_args_for_cell(manifest, cell)
        remote_cmd = _build_sbatch_remote_command(
            sbatch_flags=sbatch_flags,
            pipeline_args=pipeline_args,
            remote_project_dir=remote_project_dir,
            remote_scratch_dir=remote_scratch_dir,
        )
        code, out, err = _ssh(host, remote_cmd, dry_run=args.dry_run)
        if code != 0:
            failures.append(f"cell {label}: ssh sbatch failed (code {code}): {err or out}")
            continue
        # `sbatch --parsable` prints `<jobid>` or `<jobid>;<cluster>`.
        job_id = out.split(";")[0].strip() or "unknown"
        submitted.append(
            {
                "label": label,
                "cell": cell,
                "args": pipeline_args,
                "job_id": job_id,
                "job_name": job_name,
                "log_out": f"{log_dir_remote}/{job_name}-{job_id}.out",
                "log_err": f"{log_dir_remote}/{job_name}-{job_id}.err",
            }
        )
        print(f"submitted {job_id}  job-name={job_name}  args={' '.join(pipeline_args)}")

    if not submitted and failures:
        for f in failures:
            print(f"  FAILURE: {f}", file=sys.stderr)
        sys.exit("submit.py: no jobs were submitted; aborting (no experiments.md entry written).")

    # Always write a journal entry even if some cells failed, so partial submissions are recorded.
    entry_md = _format_experiment_entry(
        manifest=manifest,
        manifest_path=manifest_path,
        git_state=git_state,
        host=host,
        remote_project_dir=remote_project_dir,
        remote_scratch_dir=remote_scratch_dir,
        submitted_cells=submitted,
    )
    if not args.dry_run:
        _append_experiments_entry(entry_md)
        record_path = _write_run_record(
            {
                "submitted_at": datetime.now(timezone.utc).isoformat(),
                "manifest": manifest,
                "manifest_path": str(manifest_path.relative_to(REPO_ROOT)),
                "git": git_state,
                "remote": {"host": host, "project_dir": remote_project_dir, "scratch_dir": remote_scratch_dir},
                "cells": submitted,
                "failures": failures,
            }
        )
        print(f"\nLogged: remote_jobs/experiments.md (top entry)")
        print(f"        {record_path.relative_to(REPO_ROOT)}")
    else:
        print("\n[dry-run] would append to remote_jobs/experiments.md:\n")
        print(entry_md)

    if failures:
        print("\nSome cells failed:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
