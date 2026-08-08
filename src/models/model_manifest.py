"""Reading ``models_manifest.yaml`` — the one parser for a directory of models.

A directory of PyMC cognitive models is *defined* by its manifest, not by which
``.py`` files happen to sit in it: an ordered list of entries, each naming a
model (``<name>.py`` beside the manifest) and stating the one-sentence
hypothesis it implements (``rationale``). Enumerating the directory instead
would resurrect superseded or archived models.

Every consumer that needs "which models are in this directory" — outer-loop
seeding and carry-forward, the inner-loop zoo, recovery harnesses, model
comparison, EIG stimulus design — goes through this module, so the file format
lives in exactly one place. Malformed manifests raise: a model that silently
vanishes from a model set is a comparison run over the wrong hypotheses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

import yaml

MANIFEST_FILENAME = "models_manifest.yaml"

PathLike = Union[str, Path]


def manifest_path(models_dir: PathLike) -> Path:
    """Path of the manifest that defines ``models_dir``'s model set."""
    return Path(models_dir) / MANIFEST_FILENAME


def read_manifest_entries(
    models_dir: PathLike, *, missing_ok: bool = False
) -> List[Dict[str, str]]:
    """Ordered manifest entries for ``models_dir``, each normalised to a dict.

    An entry is a mapping with at least a ``name``; a bare string entry (older
    hand-written manifests) becomes ``{"name": <string>}``. An empty ``models``
    list is allowed — a freshly created inner-loop zoo has one — so callers
    that require a non-empty model set must say so themselves.

    Raises ``FileNotFoundError`` when the manifest is absent (unless
    ``missing_ok``, which is for directories the pipeline is still filling in)
    and ``ValueError`` when it is unreadable or an entry has no name.
    """
    path = manifest_path(models_dir)
    if not path.exists():
        if missing_ok:
            return []
        raise FileNotFoundError(f"{MANIFEST_FILENAME} not found at {path}")

    try:
        manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {MANIFEST_FILENAME} at {path}: {exc}") from exc
    if manifest is None:
        manifest = {}
    if not isinstance(manifest, dict):
        raise ValueError(f"{path} must be a mapping with a 'models' key")

    entries: List[Dict[str, str]] = []
    for entry in manifest.get("models") or []:
        normalised = entry if isinstance(entry, dict) else {"name": entry}
        if not normalised.get("name"):
            raise ValueError(f"{path} has a model entry with no name: {entry!r}")
        entries.append(normalised)
    return entries


def read_manifest_names(
    models_dir: PathLike, *, missing_ok: bool = False
) -> List[str]:
    """The model names ``models_dir``'s manifest lists, in manifest order."""
    return [entry["name"] for entry in read_manifest_entries(models_dir, missing_ok=missing_ok)]


def read_loadable_model_names(models_dir: PathLike) -> List[str]:
    """Manifest names whose ``<name>.py`` actually exists in ``models_dir``.

    Fitting and comparison need the implementation, not just the name, so a
    listed-but-missing model is skipped here. Callers check the result for
    emptiness — an empty model set is what they must refuse.
    """
    models_dir = Path(models_dir)
    return [
        name
        for name in read_manifest_names(models_dir)
        if (models_dir / f"{name}.py").exists()
    ]
