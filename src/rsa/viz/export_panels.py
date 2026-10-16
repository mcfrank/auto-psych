"""Per-display choice counts and model predictions, for the model-vs-data figures.

    python -m src.rsa.viz.export_panels --pooled <pooled.csv> --prereg <prereg modelled.csv> \
        --cache <fit cache> --models c0=<file> promoted=<file> ... --out panels.json

Every trial of the pooled data (the literature's plain trials and the campaign's live
experiments) and of the pre-registered experiment is grouped by dataset and display
(`src.rsa.context.model_key`). Each display gets its choice counts per class and each
model's posterior-mean class probabilities from its fit on the pooled data (the fit the
frozen predictions and the consolidation used; `loop_fit`, cached). The output holds
aggregates only: no trial rows, no participant ids.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import tyro

MAX_DRAWS = 100


@dataclass
class Args:
    pooled: Path
    prereg: Path
    cache: Path
    models: List[str]
    """``<label>=<model file>``, each fitted on ``pooled`` (cached)."""
    out: Path
    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 4
    seed: int = 0


def dataset_of(row) -> str:
    """'literature', 'campaign c<k>' or 'prereg' (set by the caller)."""
    if str(row["source"]) != "auto_psych":
        return "literature"
    pid = str(row["participant_id"])
    chain = pid.split(":")[1] if pid.startswith("auto_psych:c") else "c?"
    return f"campaign {chain}"


def main(args: Args) -> dict:
    from src.rsa.context import model_key
    from src.rsa.dataset import load_forced_choice
    from src.rsa.fit import FitSettings, mean_probs, posterior_flat
    from src.rsa.loop.fitting import NO_TIME_LIMIT, describe, loop_fit
    from src.rsa.model_file import RSAModel

    settings = FitSettings(num_warmup=args.num_warmup, num_samples=args.num_samples, num_chains=args.num_chains,
                           seed=args.seed, dense_mass=True)
    groups = {}  # (dataset, source, experiment, key) -> dict
    for path, which in ((args.pooled, None), (args.prereg, "prereg")):
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
    keys = list(groups)
    contexts = [groups[k]["ctx"] for k in keys]
    preds, fits = {}, {}
    for spec in args.models:
        label, _, file = spec.partition("=")
        path = Path(file)
        fitted = loop_fit(path, path.stem, args.pooled, settings, args.cache, time_limit_sec=NO_TIME_LIMIT)
        means = mean_probs(RSAModel(path, name=path.stem), posterior_flat(fitted), contexts, MAX_DRAWS, by_class=True)
        preds[label] = means
        fits[label] = dict(model=path.stem, fit=describe(fitted), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        print(f"  {label}: {path.stem} {describe(fitted)}", flush=True)
    panels = []
    for i, k in enumerate(keys):
        dataset, source, experiment, _ = k
        ctx = groups[k]["ctx"]
        classes = sorted(set(ctx.choice_classes()))
        n = sum(groups[k]["counts"].values())
        panels.append(dict(
            dataset=dataset, source=source, experiment=experiment, n=n,
            n_objects=len(ctx.objects), n_features=len(ctx.feature_names),
            query="prior" if ctx.utterance is None else "word",
            classes=[dict(cls=int(c), copies=int(sum(1 for x in ctx.choice_classes() if x == c)),
                          count=int(groups[k]["counts"].get(c, 0)),
                          preds={label: round(float(np.asarray(preds[label][i])[c]), 6) for label in preds})
                     for c in classes]))
    out = dict(note=("Choice counts per display and class, and each model's posterior-mean class probability from "
                     "its fit on the pooled data (src.rsa.viz.export_panels). Aggregates only."),
               pooled_sha256=hashlib.sha256(Path(args.pooled).read_bytes()).hexdigest(),
               prereg_sha256=hashlib.sha256(Path(args.prereg).read_bytes()).hexdigest(),
               models=fits, panels=panels)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    print(f"{len(panels)} panels -> {args.out}", flush=True)
    return out


if __name__ == "__main__":
    from src.rsa.cpus import pin_main_thread

    pin_main_thread()
    main(tyro.cli(Args))
