"""Leakage audit for holdout recovery runs.

Checks whether agent-written models contain traces of the held-out
ground truth: byte-identical copies, distinctive parameter names,
generating-model columns in agent-facing CSVs, and GT names in seed
manifests.
"""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set

import yaml

from src.subjective_randomness.holdout_data import (
    GENERATING_MODEL_COLUMN,
    _family_default_params,
)


def _distinctive_param_names(
    gt_model: str, gt_family_dir: Optional[Path] = None
) -> Set[str]:
    """The GT family's parameter names that other families do not share.

    An impossible ground truth has no pure-Python ``model_families`` counterpart;
    in that case there are no distinctive family params to leak, so the set is
    empty. (``beta``/``side_bias`` are shared by every family, so they are never
    distinctive anyway.) Only a *missing* family is tolerated — any other import
    error in an existing family still propagates loudly. ``gt_family_dir``
    reroutes the held-out GT to a pristine off-cwd source.
    """
    try:
        params = _family_default_params(gt_model, gt_family_dir)
    except ModuleNotFoundError:
        return set()
    return set(params) - {"beta", "side_bias"}


# Column names a candidate binds with ``pm.Data("<name>", ...)``: harness-
# provided columns, as opposed to features the model computes via its hook.
_PM_DATA_COLUMN = re.compile(r"""pm\.Data\(\s*["']([^"']+)["']""")


def _csv_header_columns(path: Path) -> List[str]:
    """The header row of a CSV as column names (``[]`` for an empty file)."""
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            return [column.strip() for column in row]
    return []


def _csvs_naming_generating_model(run_root: Path) -> List[str]:
    """Run-relative paths of CSVs whose header carries the held-out label."""
    return [
        str(path.relative_to(run_root))
        for path in sorted(run_root.rglob("*.csv"))
        if GENERATING_MODEL_COLUMN in _csv_header_columns(path)
    ]


# The loop writes a manifest per experiment under its results root, listing the
# models carried into that experiment. Those are OUTPUTS: a candidate the agent
# happened to name after the held-out model belongs in ``any_gt_named``, not in
# the manifest channel, which is about the seed catalogue shipped in the
# checkout. The results root is ``<checkout>/_runs`` in the Slurm array.
_RESULTS_DIR_NAME = "_runs"


def _manifests_naming_gt(
    checkout_root: Path, gt_model: str, *, run_root: Optional[Path] = None
) -> List[str]:
    """Checkout-relative paths of *seed* manifests that still list ``gt_model``.

    Removing the held-out ``.py`` from the agent's checkout left its *name* and
    rationale in the manifests beside it, which is the answer in plain text.
    Matched on parsed model names, so a rationale mentioning another model is
    not a false positive. Manifests the loop itself wrote (under ``run_root`` or
    any ``_runs`` tree) are skipped — see ``_RESULTS_DIR_NAME``.
    """
    named: List[str] = []
    run_root = Path(run_root).resolve() if run_root is not None else None
    for path in sorted(checkout_root.rglob("models_manifest.yaml")):
        if _RESULTS_DIR_NAME in path.parts:
            continue
        if run_root is not None and run_root in path.resolve().parents:
            continue
        try:
            manifest = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            # A malformed manifest is the model-set validator's business, not
            # this audit's; it fails loudly there in its own right.
            continue
        entries = manifest.get("models") or []
        if any(
            isinstance(entry, Mapping) and entry.get("name") == gt_model
            for entry in entries
        ):
            named.append(str(path.relative_to(checkout_root)))
    return named


def leakage_check(
    run_root: Path,
    gt_model: str,
    *,
    seed_models_dir: Path,
    n_experiments: int,
    gt_models_dir: Optional[Path] = None,
    gt_family_dir: Optional[Path] = None,
    checkout_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Audit a run for ground-truth leakage into agent-written models.

    Agents can read the project assets dir, which contains the held-out seed
    model's source. This flags (without enforcing) byte-identical copies,
    mentions of the GT family's distinctive parameter names, and files named
    after the GT model, across every experiment's ``cognitive_models/`` and
    ``model_loop/models/``. Heuristic: a paraphrased reimplementation can evade
    it, so flags are an audit trail, not proof of a clean run.

    ``gt_models_dir`` is where the ground-truth source lives for the byte-
    identity hash (default: ``seed_models_dir``). For an impossible ground truth
    it points at the impossible-models directory instead, and the source is not
    in the project assets the agents can read, so identical-copy leakage is
    effectively impossible.

    It also audits the two channels that carried the held-out model's *name*
    (the 2026-09 review panel's finding, closed at the source by
    ``strip_generating_model`` and the array's manifest scrub):

    * ``any_csv_generating_model`` — a CSV under the run tree whose header still
      carries ``generating_model``, whose value is the held-out model on every
      row. Every agent reads these files.
    * ``any_manifest_gt_named`` — a ``models_manifest.yaml`` in the agent's
      checkout still listing the held-out model by name. Needs
      ``checkout_root``; without it the channel is unchecked and the flag is
      ``None`` rather than ``False``, so an unchecked channel is never read as
      a clean one.

    Per admitted model it records ``data_columns``: the featurizer columns the
    model binds with ``pm.Data(...)``. A model assembled entirely out of
    provided columns is a regression on the harness's features rather than a
    mechanism, so a report can say per ground truth how much of recovery is
    which.
    """
    run_root = Path(run_root)
    gt_models_dir = (
        Path(gt_models_dir) if gt_models_dir is not None else Path(seed_models_dir)
    )
    gt_hash = hashlib.sha256(
        (gt_models_dir / f"{gt_model}.py").read_bytes()
    ).hexdigest()
    gt_params: Dict[str, float] = {}
    try:
        gt_params = _family_default_params(gt_model, gt_family_dir)
    except ModuleNotFoundError:
        pass
    distinctive = set(gt_params) - {"beta", "side_bias"}
    # Distinctive param VALUES an agent could paste from the leaked source. Drop
    # trivially common values (0/0.5/1) that would false-positive everywhere; a
    # pasted 4-decimal generating value is otherwise near-impossible by chance.
    distinctive_values = {
        str(gt_params[k]) for k in distinctive if gt_params[k] not in (0.0, 0.5, 1.0)
    }

    files: List[Dict[str, Any]] = []
    for exp_num in range(1, n_experiments + 1):
        exp_dir = run_root / f"experiment{exp_num}"
        for sub in ("cognitive_models", Path("model_loop") / "models"):
            model_dir = exp_dir / sub
            if not model_dir.is_dir():
                continue
            for path in sorted(model_dir.glob("*.py")):
                source = path.read_text(encoding="utf-8")
                data_columns = sorted(set(_PM_DATA_COLUMN.findall(source)))
                files.append(
                    {
                        "path": str(path.relative_to(run_root)),
                        "identical": hashlib.sha256(path.read_bytes()).hexdigest()
                        == gt_hash,
                        "mentions_gt_params": any(p in source for p in distinctive),
                        "mentions_gt_values": any(
                            v in source for v in distinctive_values
                        ),
                        "gt_named": path.name == f"{gt_model}.py",
                        "data_columns": data_columns,
                        "n_data_cols": len(data_columns),
                    }
                )
    csv_flagged = _csvs_naming_generating_model(run_root)
    manifest_flagged = (
        _manifests_naming_gt(Path(checkout_root), gt_model, run_root=run_root)
        if checkout_root is not None
        else []
    )
    return {
        "files": files,
        "any_identical": any(f["identical"] for f in files),
        "any_mention": any(f["mentions_gt_params"] for f in files),
        "any_value_mention": any(f["mentions_gt_values"] for f in files),
        "any_gt_named": any(f["gt_named"] for f in files),
        "csv_generating_model_files": csv_flagged,
        "any_csv_generating_model": bool(csv_flagged),
        "manifest_gt_named_files": manifest_flagged,
        # None, not False: nobody looked at this channel.
        "any_manifest_gt_named": (
            bool(manifest_flagged) if checkout_root is not None else None
        ),
        "max_data_cols": max((f["n_data_cols"] for f in files), default=0),
    }
