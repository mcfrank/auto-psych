"""Build a JATOS study archive (.jzip) from implementer output and import via JATOS API."""

import io
import json
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import urllib.request
from urllib import error as urllib_error


# JATOS study run URL pattern: base + /jatos/publix/{studyId}/start?batchId={batchId}&generalSingle
def build_jzip(experiment_dir: Path, study_title: str = "Auto-psych experiment") -> bytes:
    """
    Build a JATOS study archive (.jzip) from the implementer output directory.
    Returns the ZIP file as bytes.
    Structure: dirName/index.html (+ other assets), {study_uuid}.jas at root.
    """
    experiment_dir = Path(experiment_dir)
    dir_name = "study_assets"
    study_uuid = str(uuid.uuid4())
    component_uuid = str(uuid.uuid4())
    batch_uuid = str(uuid.uuid4())

    # JAS: match structure of reference (templates/jzip_contents/hello_world*.jas)
    # - study-level "active": true; null for optional strings; same batch allowedWorkerTypes
    jas = {
        "version": "3",
        "data": {
            "uuid": study_uuid,
            "title": study_title,
            "description": "",
            "active": True,
            "groupStudy": False,
            "linearStudy": False,
            "dirName": dir_name,
            "comments": None,
            "jsonData": None,
            "endRedirectUrl": None,
            "componentList": [
                {
                    "uuid": component_uuid,
                    "title": "Experiment",
                    "htmlFilePath": "index.html",
                    "reloadable": False,
                    "active": True,
                    "comments": "",
                    "jsonData": None,
                }
            ],
            "batchList": [
                {
                    "uuid": batch_uuid,
                    "title": "Default",
                    "active": True,
                    "maxActiveMembers": None,
                    "maxTotalMembers": None,
                    "maxTotalWorkers": None,
                    "allowedWorkerTypes": ["PersonalSingle", "Jatos", "PersonalMultiple"],
                    "comments": None,
                    "jsonData": None,
                }
            ],
        },
    }

    # Build asset paths in deterministic order (match reference: dir then files)
    asset_entries: list[tuple[str, str]] = []
    if (experiment_dir / "index.html").exists():
        asset_entries.append((f"{dir_name}/index.html", (experiment_dir / "index.html").read_text(encoding="utf-8")))
    if (experiment_dir / "stimuli.json").exists():
        asset_entries.append((f"{dir_name}/stimuli.json", (experiment_dir / "stimuli.json").read_text(encoding="utf-8")))
    asset_entries.sort(key=lambda e: e[0])

    # Build zip with Java-compatible metadata so JATOS (Java) accepts it.
    # Re-zipped content fails with "Study is invalid"; exact export works — difference is ZIP format.
    # Match Java ZipOutputStream: DEFLATED, create_system=0 (MS-DOS), MS-DOS external_attr, no UTF-8 flag.
    def _java_style_zinfo(arcname: str, compress: bool) -> zipfile.ZipInfo:
        z = zipfile.ZipInfo(arcname)
        z.compress_type = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
        z.create_system = 0  # MS-DOS (Java default)
        z.external_attr = 0x20 << 24  # MS-DOS: archive bit set (0x20 in high byte)
        return z

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for arcname, content in asset_entries:
            zf.writestr(_java_style_zinfo(arcname, True), content)
        jas_name = f"{dir_name}{uuid.uuid4().int % 10**19}.jas"
        jas_str = json.dumps(jas, separators=(",", ":"))
        zf.writestr(_java_style_zinfo(jas_name, True), jas_str)
    return buf.getvalue()


def import_study_to_jatos(
    base_url: str,
    token: str,
    jzip_bytes: bytes,
) -> Tuple[Optional[int], Optional[str]]:
    """
    POST the .jzip to JATOS API. Returns (study_id, error_message).
    On success error_message is None.
    """
    base_url = base_url.rstrip("/")
    url = f"{base_url}/jatos/api/v1/study"
    try:
        # multipart/form-data with field name "study"
        boundary = "----WebKitFormBoundary" + uuid.uuid4().hex[:16]
        body_start = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="study"; filename="study.jzip"\r\n'
            "Content-Type: application/zip\r\n\r\n"
        )
        body_end = f"\r\n--{boundary}--\r\n"
        body = body_start.encode("utf-8") + jzip_bytes + body_end.encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(body)),
            },
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        study_id = data.get("id")
        return (study_id, None)
    except urllib_error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8")
        except Exception:
            err_body = str(e)
        # If JSON, include full body so validation/cause details are visible
        try:
            err_json = json.loads(err_body)
            err_body = json.dumps(err_json, indent=2)
        except Exception:
            pass
        return (None, f"HTTP {e.code}: {err_body}")
    except Exception as e:
        return (None, str(e))


def get_study_properties(base_url: str, token: str, study_id: int) -> Optional[Dict[str, Any]]:
    """GET study properties including batch and component IDs."""
    base_url = base_url.rstrip("/")
    url = f"{base_url}/jatos/api/v1/studies/{study_id}/properties?withBatchProperties=true&withComponentProperties=true"
    try:
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def get_batch_and_component_ids(props: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    """Extract first batch ID and first component ID from study properties JSON."""
    # Response may have batchList/componentList at top level or under "data"
    data = props if props.get("batchList") is not None or props.get("componentList") is not None else props.get("data") or {}
    batch_list = data.get("batchList") or []
    comp_list = data.get("componentList") or []
    batch_id = batch_list[0].get("id") or batch_list[0].get("batchId") if batch_list else None
    comp_id = comp_list[0].get("id") or comp_list[0].get("componentId") if comp_list else None
    return (batch_id, comp_id)


def build_study_run_url(base_url: str, study_id: int, batch_id: int) -> str:
    """Build the URL participants use to run the study (GeneralSingle batch)."""
    base_url = base_url.rstrip("/")
    return f"{base_url}/jatos/publix/{study_id}/start?batchId={batch_id}&generalSingle"
