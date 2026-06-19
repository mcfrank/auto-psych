"""Freeze a curated set of runs into a self-contained static viewer snapshot.

The live viewer (:mod:`src.viewer.server`) is a Flask app that walks the data
tree on every request. To put the *results* on the web for collaborators we
don't need a server: the viewer's API is purely read-only, so we can call the
same scanners once per curated run and write their JSON to plain files, copy the
referenced figures, and ship the vanilla-JS frontend alongside. Firebase Hosting
then serves the directory as a static site.

Run it against the data directory::

    uv run python -m src.viewer.freeze \
        --data-root data \
        --out-dir viewer_dist \
        --run-paths outer_loop/subjective_randomness

then deploy ``viewer_dist`` to Firebase Hosting (see ``firebase.viewer.json``).

Layout produced under ``out-dir`` (mirrors the live API, but as static files —
the frontend's static mode resolves these paths when ``VIEWER_STATIC_BASE`` is
set, see ``static/app.js``)::

    index.html  app.js  styles.css            the frontend, assets relativised
    data/index.json                           the curated runs (a filtered RunIndex)
    data/run/<path>/run.json                  one run summary  (== GET /api/run/<path>)
    data/run/<path>/experiment/<unit>.json    one experiment   (== .../experiment?unit=<unit>)
    data/run/<path>/files/<rel>               figures + the deployed experiment page

Design principle (research code): publishing is explicit and loud. A run that is
not actually a run, or an empty run list, raises rather than silently producing
an empty or partial site.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import tyro
from pyprojroot import here

from src.viewer.models import RunIndex
from src.viewer.scan import scan_index, scan_run, scan_run_experiment

_STATIC_DIR = Path(__file__).parent / "static"
# Subdirectory of the snapshot holding the frozen JSON + files. The frontend's
# static mode is told this base via window.VIEWER_STATIC_BASE (see _write_frontend).
_STATIC_BASE = "data"


def _unit_slug(unit: str) -> str:
    """Map an experiment unit to a static filename stem.

    The run directory itself is unit ``"."``; ``_root`` keeps it a valid path
    segment. No real unit is named ``_root`` (units are ``experimentN`` /
    ``smoke`` / ``loop``), so there is no collision.
    """
    return "_root" if unit == "." else unit


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _copy_file(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(f"Referenced file is missing: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def _write_frontend(out_dir: Path) -> None:
    """Copy the vanilla-JS frontend and switch its shell into static mode."""
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(_STATIC_DIR / "app.js", out_dir / "app.js")
    shutil.copyfile(_STATIC_DIR / "styles.css", out_dir / "styles.css")

    html = (_STATIC_DIR / "index.html").read_text()
    # The live shell loads assets from the Flask /static mount; a static host
    # serves them next to index.html, so relativise the references.
    html = html.replace("/static/styles.css", "styles.css").replace("/static/app.js", "app.js")
    # Tell app.js to resolve API paths against the frozen files instead of Flask.
    html = html.replace(
        '<script src="app.js"></script>',
        f'<script>window.VIEWER_STATIC_BASE = "{_STATIC_BASE}";</script>\n'
        '  <script src="app.js"></script>',
    )
    (out_dir / "index.html").write_text(html)


def _freeze_run(data_root: Path, out_data: Path, run_path: str) -> tuple[int, int]:
    """Freeze one run; return (n_files_copied, n_participant_preview_rows)."""
    run = scan_run(data_root, run_path)
    run_dir = data_root / run_path
    run_out = out_data / "run" / run_path
    _write_json(run_out / "run.json", run.model_dump())

    n_files = 0
    # Run-level analysis figures (the only run-scoped files the frontend fetches).
    for rel in run.figures:
        _copy_file(run_dir / rel, run_out / "files" / rel)
        n_files += 1

    n_preview_rows = 0
    for ref in run.experiments:
        exp = scan_run_experiment(data_root, run_path, ref.unit)
        _write_json(run_out / "experiment" / f"{_unit_slug(ref.unit)}.json", exp.model_dump())
        if exp.data is not None:
            n_preview_rows += len(exp.data.rows_preview)
        # The deployed experiment page is lazy-loaded by the frontend on demand,
        # so copy it for units that actually built one.
        if exp.experiment.has_index_html:
            rel_prefix = "" if ref.unit == "." else f"{ref.unit}/"
            rel = f"{rel_prefix}experiment/index.html"
            _copy_file(run_dir / rel, run_out / "files" / rel)
            n_files += 1
    return n_files, n_preview_rows


def freeze_snapshot(data_root: Path, out_dir: Path, run_paths: list[str]) -> None:
    """Write a static snapshot of ``run_paths`` under ``out_dir``.

    Args:
        data_root: The data directory whose tree holds the runs.
        out_dir: Destination directory for the static site. Rebuilt from scratch.
        run_paths: Run paths (relative to ``data_root``) to publish. Each must be
            a run discovered by :func:`find_runs`; anything else raises loudly.
    """
    data_root = Path(data_root).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    if not run_paths:
        raise ValueError("No runs to freeze: pass at least one path via --run-paths.")

    # Validate against the runs the viewer actually discovers. This both rejects
    # non-runs and rules out prefix-nesting collisions by construction (find_runs
    # stops at each run, so no run path is a prefix of another).
    known = {ref.path: ref for ref in scan_index(data_root).runs}
    unknown = [path for path in run_paths if path not in known]
    if unknown:
        raise ValueError(
            f"Not runs under {data_root} (per find_runs): {unknown}. "
            f"Known runs: {sorted(known)}"
        )

    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_data = out_dir / _STATIC_BASE

    selected = [known[path] for path in run_paths]
    _write_json(out_data / "index.json", RunIndex(runs=selected).model_dump())

    total_files = 0
    total_preview_rows = 0
    for path in run_paths:
        n_files, n_rows = _freeze_run(data_root, out_data, path)
        total_files += n_files
        total_preview_rows += n_rows

    _write_frontend(out_dir)

    print(f"Froze {len(run_paths)} run(s) into {out_dir}:")
    for path in run_paths:
        print(f"  - {path}")
    # Publishing is to a public URL: surface that response previews and agent
    # transcripts are inlined, so the user can vet them before deploying.
    print(
        f"Copied {total_files} file(s). The JSON inlines {total_preview_rows} participant "
        "response-preview row(s) and full agent transcripts — review before deploying publicly."
    )


def main(
    data_root: Path = here() / "data",
    out_dir: Path = here() / "viewer_dist",
    run_paths: list[str] = [],  # noqa: B006 — tyro maps this to a required list CLI arg
) -> None:
    """Freeze curated runs into a static viewer snapshot for web deployment.

    Args:
        data_root: The data directory whose tree holds the runs.
        out_dir: Destination directory for the static site (rebuilt each run).
        run_paths: Run paths to publish, e.g. ``outer_loop/subjective_randomness``.
    """
    freeze_snapshot(data_root=data_root, out_dir=out_dir, run_paths=run_paths)


if __name__ == "__main__":
    tyro.cli(main)
