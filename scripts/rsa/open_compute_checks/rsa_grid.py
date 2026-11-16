"""The standard-RSA grid (src/rsa_open/grid.py) on every reference-game study: build the pooled corpus, fit one
grid theory, or report (docs/auto_open/RSA_GRID.md). No agents, no Prolific; the report is aggregates.

    python scripts/rsa/open_compute_checks/rsa_grid.py corpus --studies studies.yaml --out corpus.json
    python -m src.rsa_open.grid grid/                        # the 66 theory files and their manifest
    python scripts/rsa/open_compute_checks/rsa_grid.py fit --corpus corpus.json --grid grid --index $i --out fits
    python scripts/rsa/open_compute_checks/rsa_grid.py report --corpus corpus.json --grid grid --out fits \
        [--extra data/rsa_open/grid/discovered]               # seeds and discovered theories, fitted into fits/ too
    (fit them with `fit --grid data/rsa_open/grid/discovered --index $i --out fits`)

`studies.yaml` lists the studies (each path a copy, never the original):

    literature: [<csv>, ...]                 # forced-choice trial CSVs (src.rsa.dataset format): one block E,
                                             #   temperature fixed at 1 as in the campaign; each source a study
    closed: {<label>: <modelled.csv>, ...}   # closed-campaign experiments (same format): one choice block
    corpora:                                 # open-campaign corpus.json files: their blocks except E
      - {path: <corpus.json>, prefix: pilot1}
      - {path: <corpus.json>, prefix: s1_open}
      - {path: <corpus.json>, prefix: s1_fixed, merge: true}   # merge: the corpus's live blocks as one block

Every trial keeps its study (a block's `sources`), so scores are reported per study as well as pooled.
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import src.rsa_open  # noqa: E402,F401  (XLA's CPU pool size, before anything starts JAX)

REFERENCE = "grid_d1_nosal_bool_nocost"  # plain RSA (rsa_l1)
# pragmods' experiments that manipulate the prior (familiarization base rates, valence, colour salience, greyscale): the
# closed campaign's literature block left them out (docs/auto_rsa/LOOP_ALGORITHM.md, E), so its theories never saw
# them. Reported as their own study, `pragmods_prior`, split from each trial's unit (no refit: the folds are unchanged).
PRIOR_MANIPULATIONS = {"E5_baserate", "E6_valence", "E7_color", "color_prior_rerun"}
AXES = ("depth", "salience", "kind", "semantics", "cost")


def build_corpus(studies_path: Path, out: Path) -> Path:
    import yaml

    from src.rsa.dataset import load_forced_choice
    from src.rsa.split import unit_keys
    from src.rsa_open.corpus import LITERATURE_FIXED, Block
    from src.rsa_open.links import likelihood
    from src.rsa_open.loop.fitting import read_corpus, write_corpus

    spec = yaml.safe_load(Path(studies_path).read_text())
    blocks: List[Block] = []

    def choice_block(name, frames_and_labels, fixed=()):
        contexts, responses, units, sources = [], [], [], []
        for path, label in frames_and_labels:
            t = load_forced_choice(Path(path))
            frame = t.frame
            labels = (frame["source"].astype(str).tolist() if label is None and "source" in frame.columns
                      else [label or Path(path).stem] * len(frame))
            contexts += t.contexts
            responses += [{"choice": int(c)} for c in t.choices]
            units += [f"{lab}:{u}" for lab, u in zip(labels, unit_keys(frame).astype(str))]
            sources += labels
        return Block(name, likelihood("choice"), tuple(contexts), tuple(responses), tuple(units), tuple(sources), fixed)

    if spec.get("literature"):
        blocks.append(choice_block("E", [(p, None) for p in spec["literature"]], LITERATURE_FIXED))
    if spec.get("closed"):
        blocks.append(choice_block("closed", [(p, f"closed_{label}") for label, p in spec["closed"].items()]))
    for c in spec.get("corpora", []):
        live = [b for b in read_corpus(Path(c["path"])) if b.name != "E"]
        if c.get("merge"):
            links = {json.dumps([b.link.name, b.link.options], sort_keys=True) for b in live}
            if len(links) != 1:
                raise ValueError(f"{c['path']}: merge needs one link for every block, not {sorted(links)}")
            first = live[0]
            blocks.append(Block(c["prefix"], first.link, sum((b.contexts for b in live), ()),
                                sum((b.responses for b in live), ()),
                                tuple(f"{c['prefix']}_{b.name}:{u}" for b in live for u in b.units),
                                tuple(f"{c['prefix']}_{b.name}" for b in live for _ in b.units), first.fixed))
        else:
            for b in live:
                name = f"{c['prefix']}_{b.name}"
                blocks.append(Block(name, b.link, b.contexts, b.responses, tuple(f"{name}:{u}" for u in b.units),
                                    tuple(name for _ in b.units), b.fixed))
    if not blocks:
        raise ValueError(f"{studies_path} names no study")
    write_corpus(Path(out), blocks)
    for b in blocks:
        print(f"[grid corpus] {b.name}: {b.n} trials, {len(set(b.units))} units, studies {sorted(set(b.sources))}")
    return Path(out)


def grid_names(grid_dir: Path) -> List[str]:
    import yaml

    return [e["name"] for e in yaml.safe_load((Path(grid_dir) / "models_manifest.yaml").read_text())["models"]]


def fit_one(corpus: Path, grid_dir: Path, index: int, out: Path, folds: int, warmup: int, samples: int) -> None:
    from src.rsa.fit import FitSettings
    from src.rsa_open.corpus import assign_folds, cv_corpus
    from src.rsa_open.fit import TheoryModel
    from src.rsa_open.loop.fitting import read_corpus

    name = grid_names(grid_dir)[index]
    blocks = read_corpus(corpus)
    t = time.time()
    cv = cv_corpus(TheoryModel(Path(grid_dir) / f"{name}.py", name=name), blocks,
                   {b.name: assign_folds(b, folds, seed=0) for b in blocks}, folds,
                   settings=FitSettings(num_warmup=warmup, num_samples=samples, num_chains=4, dense_mass=True, seed=0))
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{name}.npz", **{b.name: cv.pointwise[b.name] for b in blocks})
    (out / f"{name}.json").write_text(json.dumps({"name": name, "converged": bool(cv.converged),
                                                  "seconds": time.time() - t}, indent=1))
    print(f"[grid fit] {name}: converged {cv.converged}, {(time.time() - t) / 60:.1f} min", flush=True)


def study_labels(block) -> np.ndarray:
    """Each trial's study: its source, with pragmods' prior-manipulation experiments as `pragmods_prior`."""
    sources = np.asarray(b_sources(block), dtype=object)
    for i, (s, u) in enumerate(zip(sources, block.units)):
        if s == "pragmods_trials":
            parts = str(u).split(":", 1)[-1].split("|")
            if len(parts) > 1 and parts[1] in PRIOR_MANIPULATIONS:
                sources[i] = "pragmods_prior"
    return sources.astype(str)


def b_sources(block):
    return block.sources if block.sources else [block.name] * block.n


def study_scores(lpd: Dict[str, np.ndarray], blocks) -> Dict[str, tuple]:
    """study -> (per-trial lpd, unit index) over every block, each study's trials only."""
    out = {}
    for b in blocks:
        sources = study_labels(b)
        for s in sorted(set(sources)):
            m = sources == s
            out[s] = (lpd[b.name][m], np.unique(np.asarray(b.units)[m], return_inverse=True)[1])
    return out


def report(corpus: Path, grid_dir: Path, out: Path, extra=()) -> Dict:
    import yaml

    from src.models.clustered_se import cluster_dse
    from src.rsa_open.loop.fitting import read_corpus

    blocks = read_corpus(corpus)
    manifest = {e["name"]: e for e in yaml.safe_load((Path(grid_dir) / "models_manifest.yaml").read_text())["models"]}
    extras = {}
    for d in extra:  # seeds and discovered theories (discovered_set.py): scored, not in the axis contrasts
        for e in yaml.safe_load((Path(d) / "models_manifest.yaml").read_text())["models"]:
            if e["name"] in manifest or e["name"] in extras:
                raise ValueError(f"{e['name']} is named twice")
            extras[e["name"]] = e
    done_extra = [n for n in extras if (out / f"{n}.npz").exists()]
    done = [n for n in manifest if (out / f"{n}.npz").exists()]
    if REFERENCE not in done:
        raise FileNotFoundError(f"{REFERENCE} (plain RSA) has no fit: the report is relative to it")
    scores = {n: study_scores(dict(np.load(out / f"{n}.npz")), blocks) for n in done + done_extra}
    converged = {n: json.loads((out / f"{n}.json").read_text())["converged"] for n in done + done_extra}
    studies = list(scores[REFERENCE])
    result: Dict = {"studies": studies, "theories": {}, "axes": {}, "reference": REFERENCE,
                    "missing": sorted(set(manifest) - set(done)), "extra": done_extra,
                    "missing_extra": sorted(set(extras) - set(done_extra))}
    for n in done + done_extra:
        row = {"converged": converged[n], "pooled_vs_rsa": 0.0, "studies": {}}
        for s in studies:
            mine, units = scores[n][s]
            ref = scores[REFERENCE][s][0]
            d = float(mine.sum() - ref.sum())
            row["studies"][s] = {"elpd": float(mine.sum()), "vs_rsa": d,
                                 "se": float(cluster_dse(mine, ref, units)) if n != REFERENCE else 0.0}
            row["pooled_vs_rsa"] += d
        result["theories"][n] = row
    # each axis: pairs of grid points that differ only on it, the mean difference per study and pooled
    def level(e, axis):  # salience is compared with its kind: none against listener-fc, shared-singleton, ...
        return f"{e['salience']}-{e['kind']}" if axis == "salience" and e["salience"] != "none" else str(e[axis])

    for axis in AXES:
        others = [a for a in AXES if a != axis and not (axis == "salience" and a == "kind")]
        groups: Dict[tuple, Dict] = {}
        for n in done:
            e = manifest[n]
            groups.setdefault(tuple(e[a] for a in others), {})[level(e, axis)] = n
        order = ["0", "1", "2", "3", "none", "listener-fc", "listener-singleton", "shared-fc", "shared-singleton",
                 "fc", "singleton", "bool", "graded", "extension"]  # a contrast reads (change) - (baseline)
        levels = sorted({level(manifest[n], axis) for n in done}, key=lambda v: order.index(v) if v in order else 99)
        contrasts = {}
        for a, b in itertools.combinations(levels, 2):
            pairs = [(g[a], g[b]) for g in groups.values() if a in g and b in g]
            if not pairs:
                continue
            per_study = {s: float(np.mean([scores[y][s][0].sum() - scores[x][s][0].sum() for x, y in pairs]))
                         for s in studies}
            contrasts[f"{b} - {a}"] = {"pairs": len(pairs), "pooled": float(sum(per_study.values())),
                                       "studies": per_study,
                                       "wins": sum(1 for x, y in pairs if sum(scores[y][s][0].sum() for s in studies)
                                                   > sum(scores[x][s][0].sum() for s in studies))}
        result["axes"][axis] = contrasts
    (out / "grid_report.json").write_text(json.dumps(result, indent=1))
    ranked = sorted(done + done_extra, key=lambda n: -result["theories"][n]["pooled_vs_rsa"])
    lines = ["| theory | pooled vs plain RSA | " + " | ".join(studies) + " |", "|---" * (len(studies) + 2) + "|"]
    for n in ranked:
        r = result["theories"][n]
        lines.append(f"| {n}{'' if r['converged'] else ' (not converged)'} | {r['pooled_vs_rsa']:+.0f} | "
                     + " | ".join(f"{r['studies'][s]['vs_rsa']:+.0f}" + (f" ({r['studies'][s]['vs_rsa'] / r['studies'][s]['se']:+.1f})"
                                                                        if r['studies'][s]['se'] else "")
                                  for s in studies) + " |")
    lines += ["", "Axis contrasts (mean over pairs that differ only on the axis; nats, pooled over studies; wins = "
                  "pairs where the second level is better):", ""]
    for axis, cs in result["axes"].items():
        for name, c in cs.items():
            lines.append(f"- **{axis}: {name}** ({c['pairs']} pairs): {c['pooled']:+.0f} pooled, wins {c['wins']}/{c['pairs']}; "
                         + ", ".join(f"{s} {v:+.0f}" for s, v in c["studies"].items()))
    (out / "grid_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return result


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["corpus", "fit", "report"])
    ap.add_argument("--studies", type=Path)
    ap.add_argument("--corpus", type=Path)
    ap.add_argument("--grid", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--extra", type=Path, action="append", default=[],
                    help="report: a folder of further theories (discovered_set.py), fitted into the same --out")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=1000)
    a = ap.parse_args(argv)
    if a.mode == "corpus":
        build_corpus(a.studies, a.out)
    elif a.mode == "fit":
        fit_one(a.corpus, a.grid, a.index, a.out, a.folds, a.warmup, a.samples)
    else:
        report(a.corpus, a.grid, a.out, a.extra)


if __name__ == "__main__":
    main()
