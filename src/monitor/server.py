"""Live Flask server for monitoring in-progress human studies.

Run it against the data directory::

    uv run python -m src.monitor.server

then open http://localhost:8001. Discovery walks the data tree on every request,
so newly launched studies appear without a restart, and each request reads live
Firestore + Prolific state — there is no caching to go stale.

Routes
------
- ``GET /``                          the single-page dashboard
- ``GET /api/sessions``              every monitorable study, with live summaries
- ``GET /api/session/<id>``          one session's full detail (participants,
                                     choice balance, Prolific recruitment status)
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import tyro
from flask import Flask, abort, jsonify, send_from_directory
from pyprojroot import here

from src.monitor.discovery import find_monitored_sessions
from src.monitor.report import build_detail, build_summary
from src.monitor.sources import (
    LiveFirestoreSource,
    LiveProlificSource,
    MonitorSources,
)

__all__ = ["MonitorSources", "create_app", "main"]

_STATIC_DIR = Path(__file__).parent / "static"

# Hosts that keep the (unauthenticated) dashboard on the local machine only.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", ""})


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def create_app(*, data_root: Path, sources: MonitorSources) -> Flask:
    """Build the monitor app bound to a data directory and a set of sources."""
    data_root = Path(data_root).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    app = Flask(__name__, static_folder=str(_STATIC_DIR), static_url_path="/static")
    app.config["DATA_ROOT"] = data_root
    app.config["SOURCES"] = sources

    def _find_session(session_id: str):
        for session in find_monitored_sessions(data_root):
            if session.collection_session_id == session_id:
                return session
        abort(404, description=f"No monitored session: {session_id}")

    @app.get("/")
    def index_html():
        return send_from_directory(_STATIC_DIR, "index.html")

    @app.get("/api/sessions")
    def api_sessions():
        summaries = []
        for session in find_monitored_sessions(data_root):
            responses = sources.firestore.list_responses(session.collection_session_id)
            summaries.append(build_summary(session, responses).model_dump())
        return jsonify({"generated_at": _utc_now(), "sessions": summaries})

    @app.get("/api/session/<path:session_id>")
    def api_session(session_id: str):
        session = _find_session(session_id)
        responses = sources.firestore.list_responses(session.collection_session_id)
        detail = build_detail(session, responses, sources.prolific)
        return jsonify({"generated_at": _utc_now(), **detail.model_dump()})

    @app.errorhandler(404)
    def handle_404(err):
        return jsonify(error=str(getattr(err, "description", "Not found"))), 404

    return app


def _build_live_sources(firebase_project: str | None) -> MonitorSources:
    return MonitorSources(
        firestore=LiveFirestoreSource(project=firebase_project),
        prolific=LiveProlificSource(),
    )


def main(
    data_root: Path = here() / "data",
    firebase_project: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8001,
    debug: bool = False,
) -> None:
    """Launch the live study monitor.

    Args:
        data_root: The data directory whose tree holds deployment manifests.
        firebase_project: Firestore/Firebase project id to read live responses
            from. If omitted, it is inferred from the newest deployment manifest,
            falling back to Application Default Credentials' default project.
        host: Interface to bind. Use ``0.0.0.0`` to expose on the network.
        port: Port to listen on.
        debug: Enable Flask's auto-reloading debug server.
    """
    data_root = Path(data_root).resolve()
    if firebase_project is None:
        # Infer the project from the newest deployment so the common case needs
        # no flag; falls back to the ADC default project when nothing is found.
        for session in find_monitored_sessions(data_root):
            if session.firebase_project:
                firebase_project = session.firebase_project
                break

    sources = _build_live_sources(firebase_project)
    app = create_app(data_root=data_root, sources=sources)
    if host not in _LOOPBACK_HOSTS:
        print(
            f"  [WARNING] Binding to {host} exposes this UNAUTHENTICATED dashboard — "
            "including live human-subjects data (Prolific IDs, per-participant "
            "responses) — to anyone who can reach this host on the network. Use "
            "127.0.0.1 unless you intend that.",
            flush=True,
        )
    print(
        f"Monitoring studies under {data_root}"
        + (f" (Firebase project {firebase_project})" if firebase_project else "")
        + f" at http://{host}:{port}"
    )
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    tyro.cli(main)
