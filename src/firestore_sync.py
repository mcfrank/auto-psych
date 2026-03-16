"""
Sync pipeline state between local projects/ tree and Firestore (and optional GCS).

Used by the Cloud Run Job entrypoint: sync down before run_pipeline.py, sync up after.
Firestore holds project metadata, batch metadata (including job call params and codebase hash),
run metadata, and small artifacts; large blobs go to GCS when PIPELINE_GCS_BUCKET is set.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Firestore 1MB doc limit; keep under for content field
FIRESTORE_CONTENT_MAX_BYTES = 900_000

# Env for optional overrides
PIPELINE_FIRESTORE_DATABASE = os.environ.get("PIPELINE_FIRESTORE_DATABASE", "(default)")
PIPELINE_GCS_BUCKET = os.environ.get("PIPELINE_GCS_BUCKET", "")


def _get_firestore_client():
    from google.cloud import firestore

    if PIPELINE_FIRESTORE_DATABASE and PIPELINE_FIRESTORE_DATABASE != "(default)":
        return firestore.Client(database=PIPELINE_FIRESTORE_DATABASE)
    return firestore.Client()


def _get_bucket():
    if not PIPELINE_GCS_BUCKET:
        return None
    from google.cloud import storage

    return storage.Client().bucket(PIPELINE_GCS_BUCKET)


def get_latest_batch_id(project_id: str) -> Optional[str]:
    """Return the batch_id of the most recent batch for this project, or None."""
    db = _get_firestore_client()
    query = db.collection("batches").where("project_id", "==", project_id).limit(100)
    docs = list(query.stream())
    if not docs:
        return None
    # batch_id is batch_YYYYMMDD-HHMM_hash; sort descending for latest
    docs.sort(key=lambda d: d.id, reverse=True)
    return docs[0].id


def get_batch_run_ids(batch_id: str) -> List[int]:
    """Return run_ids list from the batch document."""
    db = _get_firestore_client()
    doc = db.collection("batches").document(batch_id).get()
    if not doc.exists:
        return []
    data = doc.to_dict() or {}
    run_ids = data.get("run_ids") or []
    return [int(x) for x in run_ids if isinstance(x, (int, float)) or str(x).isdigit()]


def sync_project_down(project_id: str, projects_dir: Path) -> None:
    """
    Load project doc from Firestore and write problem_definition.md and references/
    under projects_dir / project_id.
    """
    db = _get_firestore_client()
    ref = db.collection("projects").document(project_id)
    doc = ref.get()
    if not doc.exists:
        return
    data = doc.to_dict() or {}
    proj_dir = projects_dir / project_id
    proj_dir.mkdir(parents=True, exist_ok=True)

    if "problem_definition" in data and data["problem_definition"]:
        (proj_dir / "problem_definition.md").write_text(
            data["problem_definition"], encoding="utf-8"
        )

    # References: either legacy inline array or subcollection projects/{id}/references/{filename}
    ref_dir = proj_dir / "references"
    refs = data.get("references") or []
    if isinstance(refs, dict):
        refs = [{"name": k, "content": v} for k, v in refs.items()]
    if refs:
        ref_dir.mkdir(parents=True, exist_ok=True)
        bucket = _get_bucket()
        for item in refs:
            name = item.get("name") or "unnamed"
            content = item.get("content")
            gcs_uri = item.get("gcs_uri")
            if content is not None:
                (ref_dir / name).write_text(
                    content if isinstance(content, str) else str(content),
                    encoding="utf-8",
                    errors="replace",
                )
            elif gcs_uri and bucket:
                _download_gcs_uri_to_file(gcs_uri, ref_dir / name, bucket)
    # Subcollection: references stored per-doc to stay under 1MB
    ref_coll = ref.collection("references")
    for doc_snap in ref_coll.stream():
        name = doc_snap.id
        rdata = doc_snap.to_dict() or {}
        content = rdata.get("content")
        gcs_uri = rdata.get("gcs_uri")
        ref_dir.mkdir(parents=True, exist_ok=True)
        if content is not None:
            (ref_dir / name).write_text(
                content if isinstance(content, str) else str(content),
                encoding="utf-8",
                errors="replace",
            )
        elif gcs_uri and _get_bucket():
            _download_gcs_uri_to_file(gcs_uri, ref_dir / name, _get_bucket())


def _download_gcs_uri_to_file(gcs_uri: str, path: Path, bucket) -> None:
    """Download object from gs://bucket/path to path."""
    from google.cloud import storage

    # Assume gcs_uri is gs://bucket_name/path/to/object
    if not gcs_uri.startswith("gs://"):
        return
    parts = gcs_uri[5:].split("/", 1)  # drop gs://
    if len(parts) != 2:
        return
    blob = bucket.blob(parts[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(path))


def sync_batch_runs_down(
    project_id: str, batch_id: str, run_ids: List[int], projects_dir: Path
) -> None:
    """
    Load batch and its run docs from Firestore; materialize run dirs and artifacts
    under projects_dir / project_id / batches / batch_id / run{N}.
    """
    db = _get_firestore_client()
    bucket = _get_bucket()
    batch_ref = db.collection("batches").document(batch_id)
    batch_doc = batch_ref.get()
    if not batch_doc.exists:
        return
    batch_dir = projects_dir / project_id / "batches" / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    for run_id in run_ids:
        run_ref = batch_ref.collection("runs").document(str(run_id))
        run_doc = run_ref.get()
        if not run_doc.exists:
            continue
        run_dir = batch_dir / f"run{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        # Load artifacts subcollection
        for art_doc in run_ref.collection("artifacts").stream():
            slug = art_doc.id
            data = art_doc.to_dict() or {}
            content = data.get("content")
            gcs_uri = data.get("gcs_uri")
            out_path = run_dir / slug
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if content is not None:
                out_path.write_text(
                    content if isinstance(content, str) else str(content),
                    encoding="utf-8",
                    errors="replace",
                )
            elif gcs_uri and bucket:
                _download_gcs_uri_to_file(gcs_uri, out_path, bucket)


def _slug(path: Path, base: Path) -> str:
    """Relative path as Firestore-safe doc id (e.g. 1_theory/rationale.md)."""
    try:
        rel = path.relative_to(base)
        return str(rel).replace("\\", "/")
    except ValueError:
        return path.name


def _upload_run_artifacts(
    run_dir: Path,
    run_ref,
    bucket: Optional[Any],
    project_id: str,
    batch_id: str,
    run_id: int,
) -> None:
    """Walk run_dir and write each file to Firestore artifacts (or GCS + gcs_uri)."""
    for f in run_dir.rglob("*"):
        if not f.is_file():
            continue
        try:
            rel = f.relative_to(run_dir)
        except ValueError:
            continue
        slug = str(rel).replace("\\", "/")
        try:
            raw = f.read_bytes()
        except OSError:
            continue
        if len(raw) > FIRESTORE_CONTENT_MAX_BYTES and bucket:
            # Upload to GCS
            blob_path = f"projects/{project_id}/batches/{batch_id}/runs/{run_id}/{slug}"
            blob = bucket.blob(blob_path)
            blob.upload_from_string(
                raw,
                content_type="application/octet-stream"
                if raw.startswith(b"\xff")
                else "text/plain; charset=utf-8",
            )
            gcs_uri = f"gs://{bucket.name}/{blob_path}"
            run_ref.collection("artifacts").document(slug).set({"gcs_uri": gcs_uri})
        else:
            try:
                text = raw.decode("utf-8", errors="replace")
            except Exception:
                continue
            run_ref.collection("artifacts").document(slug).set({"content": text})


def sync_batch_up(
    project_id: str,
    batch_dir: Path,
    job_metadata: Dict[str, Any],
) -> None:
    """
    After pipeline run: create/update batch doc and run docs in Firestore (and GCS for
    large artifacts). job_metadata must include: mode, n_participants, max_retries,
    append, runs_spec. commit_hash and dirty are read from batch_dir/commit_hash.txt.
    """
    db = _get_firestore_client()
    bucket = _get_bucket()
    batch_id = batch_dir.name
    if not batch_id.startswith("batch_"):
        return

    # Read commit_hash.txt written by run_pipeline
    commit_hash = ""
    dirty = False
    commit_path = batch_dir / "commit_hash.txt"
    if commit_path.exists():
        text = commit_path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.startswith("commit="):
                commit_hash = line[7:].strip()
            if line.startswith("dirty="):
                dirty = line[5:].strip().lower() in ("true", "1", "yes")

    # Collect run_ids from batch dir
    run_ids: List[int] = []
    for d in batch_dir.iterdir():
        if d.is_dir() and d.name.startswith("run"):
            try:
                run_ids.append(int(d.name[3:]))
            except ValueError:
                pass
    run_ids.sort()

    from google.cloud import firestore

    batch_ref = db.collection("batches").document(batch_id)
    batch_data: Dict[str, Any] = {
        "project_id": project_id,
        "created_at": firestore.SERVER_TIMESTAMP,
        "commit_hash": commit_hash,
        "dirty": dirty,
        "run_ids": run_ids,
        "mode": job_metadata.get("mode", ""),
        "n_participants": job_metadata.get("n_participants"),
        "max_retries": job_metadata.get("max_retries"),
        "append": job_metadata.get("append", False),
        "runs_spec": job_metadata.get("runs_spec", ""),
    }
    batch_ref.set(batch_data, merge=True)

    # Correlations
    csv_path = batch_dir / "correlations.csv"
    if csv_path.exists():
        raw = csv_path.read_bytes()
        if len(raw) <= FIRESTORE_CONTENT_MAX_BYTES:
            batch_ref.set({"correlations_csv": raw.decode("utf-8", errors="replace")}, merge=True)
        elif bucket:
            blob_path = f"projects/{project_id}/batches/{batch_id}/correlations.csv"
            bucket.blob(blob_path).upload_from_string(
                raw, content_type="text/csv; charset=utf-8"
            )
            batch_ref.set(
                {"correlations_csv_gcs_uri": f"gs://{bucket.name}/{blob_path}"},
                merge=True,
            )
    plot_path = batch_dir / "correlations_by_run.png"
    if plot_path.exists() and bucket:
        blob_path = f"projects/{project_id}/batches/{batch_id}/correlations_by_run.png"
        bucket.blob(blob_path).upload_from_filename(str(plot_path))
        batch_ref.set(
            {"correlations_plot_gcs_uri": f"gs://{bucket.name}/{blob_path}"},
            merge=True,
        )

    # Run docs and artifacts
    for run_id in run_ids:
        run_path = batch_dir / f"run{run_id}"
        if not run_path.is_dir():
            continue
        run_ref = batch_ref.collection("runs").document(str(run_id))
        run_ref.set(
            {
                "project_id": project_id,
                "batch_id": batch_id,
                "run_id": run_id,
                "status": "completed",
            },
            merge=True,
        )
        _upload_run_artifacts(
            run_path, run_ref, bucket, project_id, batch_id, run_id
        )


def _safe_ref_doc_id(filename: str) -> str:
    """Firestore document IDs cannot contain '/'; use filename, sanitized."""
    return filename.replace("/", "_")


def ensure_project_in_firestore(project_id: str, project_dir: Path) -> None:
    """
    One-time or migration: write local project (problem_definition.md, references/)
    into Firestore so the job can sync it down later. References are stored in
    subcollection projects/{id}/references/{filename} to stay under the 1MB doc limit.
    """
    db = _get_firestore_client()
    ref = db.collection("projects").document(project_id)
    data: Dict[str, Any] = {}
    prob = project_dir / "problem_definition.md"
    if prob.exists():
        data["problem_definition"] = prob.read_text(encoding="utf-8")
    if data:
        ref.set(data, merge=True)

    ref_dir = project_dir / "references"
    if not ref_dir.exists():
        return
    bucket = _get_bucket()
    ref_coll = ref.collection("references")
    for f in ref_dir.iterdir():
        if not f.is_file():
            continue
        try:
            raw = f.read_bytes()
        except OSError:
            continue
        try:
            content = raw.decode("utf-8", errors="replace")
        except Exception:
            continue
        doc_id = _safe_ref_doc_id(f.name)
        if len(raw) > FIRESTORE_CONTENT_MAX_BYTES and bucket:
            blob_path = f"projects/{project_id}/references/{f.name}"
            bucket.blob(blob_path).upload_from_string(
                content, content_type="text/plain; charset=utf-8"
            )
            ref_coll.document(doc_id).set({"gcs_uri": f"gs://{PIPELINE_GCS_BUCKET}/{blob_path}"})
        else:
            if len(raw) > FIRESTORE_CONTENT_MAX_BYTES:
                # No GCS: skip or store truncated (would break sync); skip to avoid 1MB error
                continue
            ref_coll.document(doc_id).set({"content": content})
