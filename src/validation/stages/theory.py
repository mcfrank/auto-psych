"""Validation for 1_theory outputs."""

from pathlib import Path
from typing import Any, Dict

import yaml

from src.models.theorist.loader import get_model_callable, get_model_names_from_manifest
from src.validation.types import Validated


def validate_theorist_output(run_dir: Path) -> Validated:
    run_dir = Path(run_dir)
    theorist_dir = run_dir / "1_theory"
    manifest_path = theorist_dir / "models_manifest.yaml"
    details: Dict[str, Any] = {}

    if not manifest_path.exists():
        return Validated(False, "models_manifest.yaml not found", {"path": str(manifest_path)})

    try:
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return Validated(False, f"Invalid YAML: {exc}", {"path": str(manifest_path)})

    if not isinstance(data, dict):
        return Validated(False, "manifest is not a dict", details)

    models = data.get("models") or []
    if not isinstance(models, list):
        return Validated(False, "manifest 'models' is not a list", details)

    names = []
    for model in models:
        if isinstance(model, dict) and "name" in model:
            names.append(model["name"])
        elif isinstance(model, str):
            names.append(model)
    details["model_names"] = names

    loadable = get_model_names_from_manifest(data, theorist_dir)
    for name in names:
        if name not in loadable:
            return Validated(
                False,
                f"Model '{name}' has no 1_theory/{name}.py (theorist must provide each model file)",
                details,
            )

    test_stimulus = ("HHTHTTHT", "HTHTHTHT")
    response_options = ["left", "right"]
    for name in names:
        try:
            fn = get_model_callable(name, theorist_dir)
            preds = fn(test_stimulus, response_options)
        except Exception as exc:
            return Validated(False, f"Model '{name}' raised: {exc}", details)
        if not isinstance(preds, dict):
            return Validated(False, f"Model '{name}' did not return a dict", details)
        for key in response_options:
            if key not in preds:
                return Validated(False, f"Model '{name}' missing key '{key}'", details)
        total = sum(preds[key] for key in response_options)
        if abs(total - 1.0) > 1e-5:
            return Validated(False, f"Model '{name}' probabilities sum to {total}", details)

    return Validated(True, "Theorist output valid; all models run and return scores", details)
