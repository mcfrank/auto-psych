#!/usr/bin/env python3
"""
Cloud Run Job entrypoint: sync down from Firestore, run run_pipeline.py, sync up.

Expects the same CLI args as run_pipeline.py (e.g. --project X --runs 3 --mode simulated_participants).
Sets PIPELINE_PROJECTS_DIR if not set so the pipeline writes to a known tree for sync-up.
"""

import os
import sys
from pathlib import Path

# Repo root and path for run_pipeline
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _parse_args():
    """Parse argv with same options as run_pipeline (minimal set needed for sync)."""
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--run", type=int, default=None)
    p.add_argument("--runs", type=str, default=None)
    p.add_argument("--append", action="store_true")
    p.add_argument("--mode", default="simulated_participants")
    p.add_argument("--n-participants", type=int, default=None)
    p.add_argument("--max-retries", type=int, default=None)
    p.add_argument("--agent", default=None)
    args, _ = p.parse_known_args()
    return args


def _latest_batch_dir_local(project_id: str, projects_dir: Path) -> Path | None:
    """Latest batch dir by name sort under projects_dir/project_id/batches."""
    bdir = projects_dir / project_id / "batches"
    if not bdir.exists():
        return None
    subdirs = [d for d in bdir.iterdir() if d.is_dir() and d.name.startswith("batch_")]
    if not subdirs:
        return None
    subdirs.sort(key=lambda d: d.name, reverse=True)
    return subdirs[0]


def main() -> int:
    args = _parse_args()
    project_id = args.project

    # Ensure projects dir is set so pipeline and sync use the same tree
    os.environ.setdefault("PIPELINE_PROJECTS_DIR", str(REPO_ROOT / "projects"))
    projects_dir = Path(os.environ["PIPELINE_PROJECTS_DIR"])
    projects_dir.mkdir(parents=True, exist_ok=True)

    from src.firestore_sync import (
        get_latest_batch_id,
        sync_batch_runs_down,
        sync_batch_up,
        sync_project_down,
    )

    # 1) Sync down: project + optional prior batch when appending
    sync_project_down(project_id, projects_dir)
    if args.append and not args.agent:
        batch_id = get_latest_batch_id(project_id)
        if batch_id:
            from src.firestore_sync import get_batch_run_ids

            run_ids = get_batch_run_ids(batch_id)
            if run_ids:
                sync_batch_runs_down(project_id, batch_id, run_ids, projects_dir)

    # 2) Run pipeline (same argv; run_pipeline.main() reads sys.argv)
    original_argv = list(sys.argv)
    sys.argv = [str(REPO_ROOT / "run_pipeline.py")] + sys.argv[1:]
    try:
        import run_pipeline

        run_pipeline.main()
    except SystemExit as e:
        sys.argv = original_argv
        return e.code if isinstance(e.code, int) else 1
    sys.argv = original_argv

    # 3) Sync up: batch and runs to Firestore (full pipeline only, not single-agent)
    if args.agent is None:
        batch_dir = _latest_batch_dir_local(project_id, projects_dir)
        if batch_dir:
            from src.config import (
                DEFAULT_MAX_VALIDATION_RETRIES,
                DEFAULT_SIMULATED_N_PARTICIPANTS,
            )

            job_metadata = {
                "mode": args.mode,
                "n_participants": args.n_participants if args.n_participants is not None else DEFAULT_SIMULATED_N_PARTICIPANTS,
                "max_retries": args.max_retries if args.max_retries is not None else DEFAULT_MAX_VALIDATION_RETRIES,
                "append": args.append,
                "runs_spec": args.runs or (str(args.run) if args.run is not None else ""),
            }
            sync_batch_up(project_id, batch_dir, job_metadata)

    return 0


if __name__ == "__main__":
    sys.exit(main())
