"""A page for the standard-RSA grid (docs/auto_open/RSA_GRID.md): the grid's scores, its axis contrasts and every
display of the corpus's choice blocks, people against any two grid theories. Reads only what `rsa_grid.py` wrote; no
fitting.

    python scripts/rsa/open_compute_checks/grid_viz.py --corpus corpus.json --grid grid --fits fits \
        [--extra data/rsa_open/grid/discovered] [--note "..."] --out data/rsa_open/grid/literature/grid_viz.html
    python scripts/rsa/open_compute_checks/grid_viz.py --rerender built.html [--note "..."] --out built.html

**Seeds and discovered theories** (`--extra`, `discovered_set.py`) are scored against the best converged grid theory,
per study, with each score marked seen or unseen from the theory's lineage, and every theory gets its code-size
complexity (`src.rsa_open.complexity`) for the fit-against-complexity panel. A non-grid theory may see more of a display
than its word structure (familiarization, valence, colour words); its prediction for a display is then the average over
the display's trials, and `variant_sd` records the largest spread of its log probability within one display and class.

**Displays.** A grid theory sees a display only through its words (the nameable features true of some object) and the
word said, so trials are grouped by held-out unit and that structure, with objects and words reordered to one
canonical form (the item's names are kept from the first trial as an example). Identical objects form one class, as in
the choice link. Each display falls inside one cross-validation fold, so a theory's out-of-fold probability of a class
people chose is exp(lpd) of those trials, exactly; a class nobody chose gets the remainder (split evenly, and marked,
when two or more classes were never chosen). Only choice blocks have display panels; every block is in the scores.

The page holds aggregates only: counts per display and class, never trial rows or participant ids.
"""

from __future__ import annotations

import fnmatch
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
SOURCE_LABELS = {"mayn_demberg_2022": "Mayn & Demberg 2022", "mayn_demberg_2023": "Mayn & Demberg 2023",
                 "mayn_demberg_2026": "Mayn & Demberg 2026", "pragmods_trials": "pragmods, plain",
                 "pragmods_prior": "pragmods, prior manipulated", "sikos_2021": "Sikos et al. 2021"}
BUNDLE_OPEN = '<script id="bundle" type="application/json">'


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


def theory_complexity(path: Path) -> Dict:
    from src.rsa_open.complexity import measure

    m = measure(path.read_text())
    if m["n_params"] is None:  # PARAMS built by code (the consolidation's make()): count by loading
        from src.rsa_open.fit import TheoryModel

        m["n_params"] = len(TheoryModel(path, name=path.stem).params)
    return m


def build(corpus: Path, grid_dir: Path, fits: Path, extra=()) -> Dict:
    import yaml

    from src.rsa_open.loop.fitting import read_corpus

    sys.path.insert(0, str(Path(__file__).parent))
    from rsa_grid import study_labels

    report = json.loads((fits / "grid_report.json").read_text())
    manifest = yaml.safe_load((grid_dir / "models_manifest.yaml").read_text())["models"]
    for e in manifest:
        e["group"], e["dir"] = "grid", grid_dir
    extras = []
    for d in extra:
        for e in yaml.safe_load((Path(d) / "models_manifest.yaml").read_text())["models"]:
            e["dir"] = Path(d)
            extras.append(e)
    done = [e for e in manifest + extras if e["name"] in report["theories"]]
    names = [e["name"] for e in done]
    grid_names = {e["name"] for e in manifest}
    lpd = {n: dict(np.load(fits / f"{n}.npz")) for n in names}
    blocks = read_corpus(corpus)
    raw = json.loads(Path(corpus).read_text())["blocks"]

    from src.models.clustered_se import cluster_dse

    ref = report.get("reference", "grid_d1_nosal_bool_nocost")
    units = np.unique(np.concatenate([[f"{b.name}:{u}" for u in b.units] for b in blocks]), return_inverse=True)[1]
    flat = {n: np.concatenate([lpd[n][b.name] for b in blocks]) for n in names}
    labels = np.concatenate([study_labels(b) for b in blocks])
    literature = set(np.concatenate([study_labels(b) for b in blocks if b.name == "E"])) if any(
        b.name == "E" for b in blocks) else set()
    # the best textbook theory: the converged grid theory with the highest pooled score
    best_grid = max((n for n in names if n in grid_names and report["theories"][n]["converged"]),
                    key=lambda n: report["theories"][n]["pooled_vs_rsa"])

    def seen(e, study):
        if e["group"] == "grid":
            return None  # textbook theories: nobody discovered them
        spec = e["seen"]
        if study in literature:
            return spec["literature"] == "all" or (spec["literature"] == "plain" and study != "pragmods_prior")
        return any(fnmatch.fnmatch(study, pat) for pat in spec["studies"])

    theories = []
    for e in done:
        n = e["name"]
        r = report["theories"][n]
        row = dict(name=n, group=e["group"], converged=r["converged"],
                   pooled=r["pooled_vs_rsa"], elpd=float(flat[n].sum()),
                   pooled_se=0.0 if n == ref else float(cluster_dse(flat[n], flat[ref], units)),
                   vs_best=float(flat[n].sum() - flat[best_grid].sum()),
                   vs_best_se=0.0 if n == best_grid else float(cluster_dse(flat[n], flat[best_grid], units)),
                   studies={}, complexity=theory_complexity(e["dir"] / f"{n}.py"))
        for s, v in r["studies"].items():
            m = labels == s
            u = np.unique(units[m], return_inverse=True)[1]
            row["studies"][s] = {"d": v["vs_rsa"], "se": v["se"], "elpd": v["elpd"],
                                 "vs_best": float(flat[n][m].sum() - flat[best_grid][m].sum()),
                                 "vs_best_se": 0.0 if n == best_grid else float(cluster_dse(flat[n][m], flat[best_grid][m], u)),
                                 "seen": seen(e, s)}
        if e["group"] == "grid":
            row.update(depth=e["depth"], salience=e["salience"], kind=e["kind"] or "", semantics=e["semantics"],
                       cost=e["cost"])
        else:
            row.update(rationale=e.get("rationale", ""), origin=e.get("origin", ""))
        theories.append(row)

    displays: Dict[tuple, Dict] = {}
    block_labels = {b.name: study_labels(b) for b in blocks}
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
                displays[gid] = dict(block=b.name, source=str(block_labels[b.name][i]),
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
    variant_sd: Dict[str, float] = {}  # a non-grid theory may see more than the word structure (familiarization,
    # valence, colour words): its prediction for a display is then the average over the display's variants
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
                    sd = float(np.std(l[m]))
                    if n in grid_names:
                        worst_sd = max(worst_sd, sd)
                    else:
                        variant_sd[n] = max(variant_sd.get(n, 0.0), sd)
                    p[c] = float(np.exp(l[m]).mean())
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
    for t in theories:
        t["variant_sd"] = round(variant_sd.get(t["name"], 0.0), 4)
    print(f"[grid viz] {len(names)} theories, {len(out)} displays "
          f"({estimated} with two or more classes nobody chose), worst within-class sd {worst_sd:.2g}")
    return dict(meta=dict(built=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                          n_done=sum(n in grid_names for n in names), n_total=len(manifest),
                          n_extra_done=sum(n not in grid_names for n in names), n_extra_total=len(extras),
                          reference=ref, best_grid=best_grid,
                          missing_extra=[dict(name=e["name"], group=e["group"]) for e in extras if e["name"] not in names],
                          trials=sum(b.n for b in blocks), units=sum(len(set(b.units)) for b in blocks),
                          missing=report["missing"]),
                studies=[{"id": s, "label": SOURCE_LABELS.get(s, s)} for s in report["studies"]],
                theories=theories, contrasts=report["axes"], displays=out)


def bundle_of(page: Path) -> Dict:
    """The data a built page embeds (to re-render it with the current template, without the fits)."""
    text = Path(page).read_text()
    i = text.index(BUNDLE_OPEN) + len(BUNDLE_OPEN)
    return json.loads(text[i:text.index("</script>", i)])


def render(bundle: Dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(bundle, separators=(",", ":")).replace("</", "<\\/")
    out.write_text(TEMPLATE.read_text().replace("__BUNDLE__", data))


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path)
    ap.add_argument("--grid", type=Path)
    ap.add_argument("--fits", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--extra", type=Path, action="append", default=[], help="seed/discovered theory folders")
    ap.add_argument("--note", action="append", default=[], help="a sentence shown at the top of the page (repeatable)")
    ap.add_argument("--rerender", type=Path, help="a built page: re-render its data with the current template "
                                                  "(no fits needed; --note replaces its notes)")
    a = ap.parse_args(argv)
    if a.rerender:
        bundle = bundle_of(a.rerender)
        for s in bundle["studies"]:
            s["label"] = SOURCE_LABELS.get(s["id"], s["label"])
    else:
        if not (a.corpus and a.grid and a.fits):
            raise SystemExit("--corpus, --grid and --fits are needed unless --rerender")
        bundle = build(a.corpus, a.grid, a.fits, a.extra)
    if a.note or not a.rerender:
        bundle["meta"]["notes"] = a.note
    render(bundle, a.out)
    print(f"[grid viz] wrote {a.out} ({a.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
