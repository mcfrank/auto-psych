"""External data sources for the monitor: Firestore and Prolific.

Both are defined as small protocols so tests can inject in-memory fakes and the
web layer never imports a network client directly. The live implementations are
thin wrappers: ``LiveFirestoreSource`` reads the responses subcollection for a
session, and ``LiveProlificSource`` reuses the existing Prolific client in
``src.runtime.prolific``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

ADC_HELP = (
    "Firestore read needs Application Default Credentials. Run: "
    "gcloud auth application-default login && "
    "gcloud auth application-default set-quota-project auto-psych-2c5da"
)


class FirestoreSource(Protocol):
    """Reads the live response documents for a collection session."""

    def list_responses(self, collection_session_id: str) -> list[tuple[str, dict[str, Any]]]:
        """Return ``(doc_id, data)`` pairs for every submitted response."""
        ...


class ProlificSource(Protocol):
    """Reads recruitment status for a Prolific study.

    Each method returns ``(data, error)``: on success ``error`` is ``None``; on
    failure ``data`` is ``None`` and ``error`` is a human-readable message. A
    monitoring dashboard stays useful when Prolific is down, so failures are
    reported rather than raised.
    """

    def study_status(self, study_id: str) -> tuple[dict[str, Any] | None, str | None]: ...

    def submission_counts(self, study_id: str) -> tuple[dict[str, Any] | None, str | None]: ...


@dataclass
class MonitorSources:
    """The external sources the monitor reads from."""

    firestore: FirestoreSource
    prolific: ProlificSource | None = None


class LiveFirestoreSource:
    """Reads participant responses from a real Firestore project."""

    def __init__(self, project: str | None = None):
        self._project = project
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        from google.auth.exceptions import DefaultCredentialsError
        from google.cloud import firestore

        try:
            self._client = firestore.Client(project=self._project)
        except DefaultCredentialsError as exc:
            raise RuntimeError(ADC_HELP) from exc
        return self._client

    def list_responses(self, collection_session_id: str) -> list[tuple[str, dict[str, Any]]]:
        client = self._get_client()
        responses = (
            client.collection("collection_sessions")
            .document(collection_session_id)
            .collection("responses")
        )
        return [(doc.id, doc.to_dict() or {}) for doc in responses.stream()]


class LiveProlificSource:
    """Reads study status and submission counts via the Prolific API client.

    ``src.runtime.prolific`` reports API failures through its ``(value, error)``
    return but *raises* on a missing ``PROLIFIC_API_TOKEN``, so that a
    two-hour collection poll cannot mistake a config error for a flaky
    endpoint. The monitor's contract is the opposite: it must stay up and show
    that recruitment status is unavailable. That single translation happens
    here, explicitly, rather than at every call site.
    """

    def study_status(self, study_id: str) -> tuple[dict[str, Any] | None, str | None]:
        from src.runtime.prolific import get_study

        try:
            return get_study(study_id)
        except ValueError as exc:
            return (None, f"Prolific is not configured: {exc}")

    def submission_counts(self, study_id: str) -> tuple[dict[str, Any] | None, str | None]:
        from src.runtime.prolific import get_submission_counts

        try:
            return get_submission_counts(study_id)
        except ValueError as exc:
            return (None, f"Prolific is not configured: {exc}")
