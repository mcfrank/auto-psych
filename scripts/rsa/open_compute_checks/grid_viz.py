"""A page for the standard-RSA grid (docs/auto_open/RSA_GRID.md): the grid's scores, its axis contrasts and every
display of the corpus's choice blocks, people against any two grid theories. Reads only what `rsa_grid.py` wrote; no
fitting.

    python scripts/rsa/open_compute_checks/grid_viz.py --corpus corpus.json --grid grid --fits fits \
        --out data/rsa_open/grid/literature/grid_viz.html

**Displays.** A grid theory sees a display only through its words (the nameable features true of some object) and the
word said, so trials are grouped by held-out unit and that structure, with objects and words reordered to one
canonical form (the item's names are kept from the first trial as an example). Identical objects form one class, as in
the choice link. Each display falls inside one cross-validation fold, so a theory's out-of-fold probability of a class
people chose is exp(lpd) of those trials, exactly; a class nobody chose gets the remainder (split evenly, and marked,
when two or more classes were never chosen). Only choice blocks have display panels; every block is in the scores.

The page holds aggregates only: counts per display and class, never trial rows or participant ids.
"""

from __future__ import annotations

import itertools
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
TEMPLATE = Path(__file__).with_name("grid_viz_page.html")
SOURCE_LABELS = {"mayn_demberg_2026": "Mayn & Demberg 2026", "pragmods_trials": "pragmods",
                 "sikos_2021": "Sikos et al. 2021"}


def words_of(scene: Dict) -> List[int]:
    """The nameable features true of some object, in index order (`RSAContext.present_features`)."""
    n = len(scene["feature_names"])
    nameable = range(n) if scene.get("messages") is None else sorted(scene["messages"])
    return [f for f in nameable if any(o[f] for o in scene["objects"])]


def canonical(scene: Dict):
    """(key, object order, word order) of the display's structure: the word x object matrix with the word said
    marked, minimal over object orders (words sorted within each order, the word said first)."""
    words = words_of(scene)
    objs = scene["objects"]
    said = scene["utterance"]
    best = None
    for perm in itertools.permutations(range(len(objs))):
        rows = sorted(((0 if w == said else 1, tuple(objs[o][w] for o in perm)), w) for w in words)
        key = (said is None, tuple(r for r, _ in rows))
        if best is None or key < best[0]:
            best = (key, perm, [w for _, w in rows])
    return best


def build(corpus: Path, grid_dir: Path, fits: Path) -> Dict:
    import yaml

    from src.rsa_open.loop.fitting import read_corpus

    report = json.loads((fits / "grid_report.json").read_text())
    manifest = yaml.safe_load((grid_dir / "models_manifest.yaml").read_text())["models"]
    done = [e for e in manifest if e["name"] in report["theories"]]
    names = [e["name"] for e in done]
    lpd = {n: dict(np.load(fits / f"{n}.npz")) for n in names}
    blocks = read_corpus(corpus)
    raw = json.loads(Path(corpus).read_text())["blocks"]

    from src.models.clustered_se import cluster_dse

    ref = report.get("reference", "grid_d1_nosal_bool_nocost")
    units = np.unique(np.concatenate([[f"{b.name}:{u}" for u in b.units] for b in blocks]), return_inverse=True)[1]
    flat = {n: np.concatenate([lpd[n][b.name] for b in blocks]) for n in names}
    theories = []
    for e in done:
        r = report["theories"][e["name"]]
        theories.append(dict(name=e["name"], depth=e["depth"], salience=e["salience"], kind=e["kind"] or "",
                             semantics=e["semantics"], cost=e["cost"], converged=r["converged"],
                             n_params=(e["depth"] >= 1) + (e["salience"] != "none") + (e["cost"] != "none"),
                             pooled=r["pooled_vs_rsa"], elpd=float(flat[e["name"]].sum()),
                             pooled_se=0.0 if e["name"] == ref else float(cluster_dse(flat[e["name"]], flat[ref], units)),
                             studies={s: {"d": v["vs_rsa"], "se": v["se"], "elpd": v["elpd"]}
                                      for s, v in r["studies"].items()}))

    displays: Dict[tuple, Dict] = {}
    for b, rb in zip(blocks, raw):
        if b.link.name != "choice":
            continue
        for i, t in enumerate(rb["trials"]):
            scene = t["scene"]
            key, perm, word_order = canonical(scene)
            gid = (b.name, t["unit"], key)
            cols = [tuple(scene["objects"][o][w] for w in word_order) for o in perm]
            classes = sorted(set(cols), key=cols.index)
            if gid not in displays:
                parts = t["unit"].split(":", 1)[-1].split("|")
                displays[gid] = dict(block=b.name, source=t["source"],
                                     unit=" · ".join(x.replace("_", " ") for x in parts[1:3]) or parts[0],
                                     words=[scene["feature_names"][w] for w in word_order],
                                     said=None if scene["utterance"] is None else 0, item=scene.get("item") or "",
                                     classes=[{"features": list(c), "copies": cols.count(c), "count": 0}
                                              for c in classes],
                                     trials=[], chosen=[])
            d = displays[gid]
            cls = classes.index(tuple(scene["objects"][t["response"]["choice"]][w] for w in word_order))
            d["classes"][cls]["count"] += 1
            d["trials"].append(i)
            d["chosen"].append(cls)

    out = []
    worst_sd, estimated = 0.0, 0
    for gid, d in displays.items():
        rows, chosen = np.asarray(d.pop("trials")), np.asarray(d.pop("chosen"))
        k = len(d["classes"])
        preds, totals = [], []
        for n in names:
            l = lpd[n][d["block"]][rows]
            p = np.full(k, np.nan)
            for c in range(k):
                m = chosen == c
                if m.any():
                    worst_sd = max(worst_sd, float(np.std(l[m])))
                    p[c] = float(np.exp(l[m].mean()))
            missing = np.isnan(p)
            if missing.any():
                p[missing] = max(0.0, 1.0 - np.nansum(p)) / missing.sum()
            preds.append([round(float(v), 4) for v in p])
            totals.append(round(float(l.sum()), 3))
        never = sum(c["count"] == 0 for c in d["classes"])
        estimated += never >= 2
        d.update(n=int(len(rows)), preds=preds, lpd=totals, estimated=never >= 2)
        out.append(d)
    if worst_sd > 1e-3:
        raise ValueError(f"trials of one display and class disagree on a theory's probability (sd {worst_sd:.3g}): "
                         "the grouping is not what the theories see")
    out.sort(key=lambda d: (d["source"], d["unit"], -d["n"]))
    for j, d in enumerate(out):
        d["id"] = j
    print(f"[grid viz] {len(names)} theories, {len(out)} displays "
          f"({estimated} with two or more classes nobody chose), worst within-class sd {worst_sd:.2g}")
    total = len(manifest)
    return dict(meta=dict(built=datetime.now(timezone.utc).strftime("%Y-%m-%d"), n_done=len(names), n_total=total,
                          reference=ref,
                          trials=sum(b.n for b in blocks), units=sum(len(set(b.units)) for b in blocks),
                          missing=report["missing"]),
                studies=[{"id": s, "label": SOURCE_LABELS.get(s, s)} for s in report["studies"]],
                theories=theories, contrasts=report["axes"], displays=out)


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--grid", type=Path, required=True)
    ap.add_argument("--fits", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    bundle = build(a.corpus, a.grid, a.fits)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(bundle, separators=(",", ":")).replace("</", "<\\/")
    a.out.write_text(TEMPLATE.read_text().replace("__BUNDLE__", data))
    print(f"[grid viz] wrote {a.out} ({a.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
