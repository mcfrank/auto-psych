"""Live Flask server for exploring automated-psychology runs in the browser.

Run it against the data directory::

    uv run python -m src.viewer.server

then open http://localhost:8000. The server walks the data tree on every
request, so it always reflects the latest runs — no build step. A *run* is any
directory holding experiments (or a bare model loop), identified by its path
relative to the data root.

Routes
------
- ``GET /``                                         the single-page app
- ``GET /api/index``                                every run found under the data root
- ``GET /api/run/<path>``                           one run's summary (its experiment units)
- ``GET /api/run/<path>/experiment?unit=<u>``       one experiment unit's full payload
- ``GET /api/run/<path>/files/<rel>``               a file inside the run (figures, etc.)
"""

from __future__ import annotations

from pathlib import Path

import tyro
from flask import Flask, abort, jsonify, request, send_from_directory
from pyprojroot import here

from src.viewer.scan import scan_index, scan_run, scan_run_experiment

_STATIC_DIR = Path(__file__).parent / "static"

# The /files/ route exists only to serve run-level analysis figures and the
# deployed experiment page (plus that page's own local assets). Restrict it to
# those types so it can never hand out agent transcripts (*.jsonl), raw model
# source (*.py), deployment configs/manifests (*.json/*.yaml), saved fits (*.nc),
# or anything else that happens to sit in a run directory.
_SERVABLE_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".html", ".htm", ".css", ".js"}
)

# Hosts that keep the (unauthenticated) server on the local machine only.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", ""})


def create_app(data_root: Path) -> Flask:
    """Build the Flask app bound to a data directory whose tree holds runs."""
    data_root = Path(data_root).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    def _run_dir(run_path: str) -> Path:
        """Resolve a run path under the data root, blocking traversal escapes."""
        resolved = (data_root / run_path).resolve()
        if resolved != data_root and data_root not in resolved.parents:
            abort(404, description=f"Path escapes data root: {run_path}")
        if not resolved.is_dir():
            abort(404, description=f"No such run: {run_path}")
        return resolved

    app = Flask(__name__, static_folder=str(_STATIC_DIR), static_url_path="/static")
    app.config["DATA_ROOT"] = data_root

    @app.get("/")
    def index_html():
        return send_from_directory(_STATIC_DIR, "index.html")

    @app.get("/api/index")
    def api_index():
        return jsonify(scan_index(data_root).model_dump())

    @app.get("/api/run/<path:run_path>")
    def api_run(run_path: str):
        _run_dir(run_path)
        try:
            return jsonify(scan_run(data_root, run_path).model_dump())
        except FileNotFoundError:
            abort(404, description=f"No such run: {run_path}")

    @app.get("/api/run/<path:run_path>/experiment")
    def api_run_experiment(run_path: str):
        _run_dir(run_path)
        unit = request.args.get("unit", ".")
        try:
            return jsonify(scan_run_experiment(data_root, run_path, unit).model_dump())
        except FileNotFoundError:
            abort(404, description=f"No such experiment unit: {run_path}/{unit}")

    @app.get("/api/run/<path:run_path>/files/<path:rel>")
    def api_run_file(run_path: str, rel: str):
        run_dir = _run_dir(run_path)
        # Check the type BEFORE touching the filesystem so a disallowed request
        # (e.g. a *.py model or *.jsonl transcript) is rejected without revealing
        # whether the file exists.
        if Path(rel).suffix.lower() not in _SERVABLE_SUFFIXES:
            abort(403, description=f"File type not served: {rel}")
        if not (run_dir / rel).is_file():
            abort(404, description=f"No such file: {run_path}/{rel}")
        # send_from_directory blocks path traversal outside run_dir.
        return send_from_directory(run_dir, rel)

    @app.errorhandler(403)
    def handle_403(err):
        return jsonify(error=str(getattr(err, "description", "Forbidden"))), 403

    @app.errorhandler(404)
    def handle_404(err):
        return jsonify(error=str(getattr(err, "description", "Not found"))), 404

    @app.errorhandler(ValueError)
    def handle_corrupt_data(err):
        # Corrupt artifacts must surface loudly, not be silently swallowed.
        return jsonify(error=str(err)), 500

    return app


def main(
    data_root: Path = here() / "data",
    host: str = "127.0.0.1",
    port: int = 8000,
    debug: bool = False,
) -> None:
    """Launch the run explorer.

    Args:
        data_root: The data directory whose tree holds the runs to explore.
        host: Interface to bind. Use ``0.0.0.0`` to expose on the network.
        port: Port to listen on.
        debug: Enable Flask's auto-reloading debug server.
    """
    app = create_app(data_root=data_root)
    if host not in _LOOPBACK_HOSTS:
        print(
            f"  [WARNING] Binding to {host} exposes this UNAUTHENTICATED viewer — "
            "including raw run artifacts under the data root — to anyone who can "
            "reach this host on the network. Use 127.0.0.1 unless you intend that.",
            flush=True,
        )
    print(f"Serving run explorer for {data_root} at http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    tyro.cli(main)
