"""Reference games: what each display of ``data/rsa/viz/panels.json`` showed.

``panels.json`` (``src.rsa.viz.export_panels``) holds each display's choice counts
and model predictions but not the display itself. This repeats only the grouping
half of ``export_panels.main`` (no fitting) on the same two trial files and writes,
for each display in the same order, its contents (objects, features, utterance,
...) and its choice counts per class. Stimulus structure and aggregate counts
only: no trial rows, no participant ids.

Before writing it checks alignment with ``panels.json`` and raises on any
mismatch: the input files' sha256, the number of displays, and each display's
dataset, source, experiment, trial count and per-class counts.

    python -m scripts.paper.rsa_panel_displays \
        --pooled $SCRATCH/auto-psych/rsa_prereg/pool/pooled.csv \
        --prereg $SCRATCH/auto-psych/rsa_prereg/cells/prereg/run/experiment1/data/modelled.csv
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import tyro
from pyprojroot import here

from src.rsa.viz.export_panels import dataset_of

VIZ_DIR = here() / "data" / "rsa" / "viz"


@dataclass
class Args:
    pooled: Path
    """The pooled trials ``panels.json`` was built from (its ``pooled_sha256``)."""
    prereg: Path
    """The pre-registered experiment's trials (its ``prereg_sha256``)."""
    panels: Path = VIZ_DIR / "panels.json"
    out: Path = VIZ_DIR / "panel_displays.json"


def group_displays(pooled: Path, prereg: Path) -> list[dict]:
    """One entry per display, grouped and ordered exactly as ``export_panels.main``."""
    from src.rsa.context import model_key
    from src.rsa.dataset import load_forced_choice

    groups = {}  # (dataset, source, experiment, key) -> dict
    for path, which in ((pooled, None), (prereg, "prereg")):
        trials = load_forced_choice(path)
        for (_, row), ctx, choice in zip(trials.frame.iterrows(), trials.contexts, trials.choices):
            if str(row.get("condition", "")) == "catch":
                continue
            dataset = which or dataset_of(row)
            source = "prereg" if which else str(row["source"])
            experiment = "prereg" if which else str(row["experiment"])
            k = (dataset, source, experiment, model_key(ctx))
            g = groups.setdefault(k, dict(ctx=ctx, counts={}))
            cls = ctx.choice_classes()[choice]
            g["counts"][cls] = g["counts"].get(cls, 0) + 1

    entries = []
    for (dataset, source, experiment, _), g in groups.items():
        ctx = g["ctx"]
        entries.append(dict(
            dataset=dataset, source=source, experiment=experiment, n=sum(g["counts"].values()),
            objects=[list(row) for row in ctx.objects],
            feature_names=list(ctx.feature_names),
            utterance=ctx.utterance,
            messages=None if ctx.messages is None else list(ctx.messages),
            familiarization=None if ctx.familiarization is None else list(ctx.familiarization),
            grayscale=None if ctx.grayscale is None else list(ctx.grayscale),
            valence=ctx.valence,
            nameable=list(ctx.nameable),
            choice_classes=list(ctx.choice_classes()),
            class_counts={str(c): g["counts"].get(c, 0) for c in sorted(set(ctx.choice_classes()))}))
    return entries


def check_alignment(entries: list[dict], panels_doc: dict, pooled: Path, prereg: Path) -> None:
    """Raise unless ``entries`` describe ``panels_doc["panels"]`` one for one."""
    for name, path in (("pooled", pooled), ("prereg", prereg)):
        sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if sha != panels_doc[f"{name}_sha256"]:
            raise ValueError(f"{path}: sha256 {sha} is not panels.json's {name}_sha256 "
                             f"{panels_doc[f'{name}_sha256']}")
    panels = panels_doc["panels"]
    if len(entries) != len(panels):
        raise ValueError(f"{len(entries)} displays, but panels.json has {len(panels)}")
    for i, (entry, panel) in enumerate(zip(entries, panels)):
        for field in ("dataset", "source", "experiment", "n"):
            if entry[field] != panel[field]:
                raise ValueError(f"display {i}: {field} {entry[field]!r} != panels.json's {panel[field]!r}")
        panel_counts = {str(c["cls"]): c["count"] for c in panel["classes"]}
        if entry["class_counts"] != panel_counts:
            raise ValueError(f"display {i}: class counts {entry['class_counts']} != panels.json's {panel_counts}")


def main(args: Args) -> list[dict]:
    panels_doc = json.loads(Path(args.panels).read_text())
    entries = group_displays(args.pooled, args.prereg)
    check_alignment(entries, panels_doc, args.pooled, args.prereg)
    Path(args.out).write_text(json.dumps(entries))
    print(f"{len(entries)} displays aligned with {args.panels} -> {args.out}", flush=True)
    return entries


if __name__ == "__main__":
    main(tyro.cli(Args))
