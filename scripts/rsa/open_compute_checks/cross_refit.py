"""Every theory on every paradigm of stage 1: a dry run of the bakeoff's fit A on the pilot's data.

Stage 1 (docs/auto_open/STAGE1_REPORT.md) ran an open and a fixed chain, three experiments each. Each chain's loop
only ever scored its own theories on its own blocks. Here the two chains' final corpora are merged (the
literature block E once, then open_exp1-3 and fixed_exp1-3) and each theory is fitted to all seven blocks jointly,
with grouped cross-validation within every block (whole units per fold, as the loop does). It answers, for the
pilot only and descriptively: do the fixed arm's theories hold up on the open arm's paradigms (rating,
production), do the open arm's hold up on the fixed paradigm, and how do both compare with plain RSA, the seeds and
the closed campaign's references.

    # one theory (a line of the theories file: <name>\\t<path to its .py>)
    python scripts/rsa/open_compute_checks/cross_refit.py fit --open-corpus <open>/experiment3/corpus.json \\
        --fixed-corpus <fixed>/experiment3/corpus.json --theories theories.tsv --index 0 --out <dir>
    # the table, once every theory is done
    python scripts/rsa/open_compute_checks/cross_refit.py report --open-corpus ... --fixed-corpus ... \\
        --theories theories.tsv --out <dir>

`fit` writes <out>/<name>.npz (each block's out-of-fold lpd per trial: scores, kept on Sherlock with the run) and
<name>.json (each block's elpd, convergence, time). `report` writes <out>/cross_refit.json and .md: per theory and
block, elpd over chance and over plain RSA with unit-clustered SEs, and per-display sums of the gains (aggregates,
safe to bring back). No agents, no Prolific.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import src.rsa_open  # noqa: E402,F401  (XLA's CPU pool size, before anything starts JAX)

CHANCE, RSA = "chance", "rsa_l1"


def merged_blocks(open_corpus: Path, fixed_corpus: Path):
    """E once (it must be the same in both corpora), then each chain's live blocks, renamed <arm>_<block>, their
    units prefixed so that the two chains' units never coincide."""
    from src.rsa_open.corpus import Block
    from src.rsa_open.loop.fitting import read_corpus

    out, seen_e = [], None
    for arm, path in (("open", open_corpus), ("fixed", fixed_corpus)):
        for b in read_corpus(path):
            if b.name == "E":
                key = hashlib.sha256(json.dumps([list(b.units), [dict(r) for r in b.responses]]).encode()).hexdigest()
                if seen_e is None:
                    seen_e = key
                    out.append(b)
                elif key != seen_e:
                    raise ValueError("the two corpora's literature blocks differ: they must be the same E")
                continue
            name = f"{arm}_{b.name}"
            out.append(Block(name, b.link, b.contexts, b.responses, tuple(f"{name}:{u}" for u in b.units), b.sources,
                             b.fixed))
    if seen_e is None:
        raise ValueError("no literature block E in either corpus")
    return out


def theories(path: Path) -> List[Tuple[str, Path]]:
    rows = [line.split("\t") for line in Path(path).read_text().splitlines() if line.strip()]
    return [(name, Path(p)) for name, p in rows]


def fit(args) -> None:
    from src.rsa.fit import FitSettings
    from src.rsa_open.corpus import assign_folds, cv_corpus
    from src.rsa_open.fit import TheoryModel

    name, path = theories(args.theories)[args.index]
    blocks = merged_blocks(args.open_corpus, args.fixed_corpus)
    folds = {b.name: assign_folds(b, args.folds, seed=0) for b in blocks}
    settings = FitSettings(num_warmup=args.warmup, num_samples=args.samples, num_chains=4, dense_mass=True, seed=0)
    t = time.time()
    cv = cv_corpus(TheoryModel(path, name=name), blocks, folds, args.folds, settings=settings)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / f"{name}.npz", **{b.name: cv.pointwise[b.name] for b in blocks})
    record = {"name": name, "path": str(path), "converged": bool(cv.converged), "seconds": time.time() - t,
              "elpd": {b.name: float(cv.pointwise[b.name].sum()) for b in blocks}, "settings": vars(settings)}
    (args.out / f"{name}.json").write_text(json.dumps(record, indent=1, default=str))
    print(f"[cross refit] {name}: converged {cv.converged}, {record['seconds'] / 60:.1f} min", flush=True)


def report(args) -> Dict:
    from src.models.clustered_se import cluster_dse

    blocks = merged_blocks(args.open_corpus, args.fixed_corpus)
    names = [n for n, _ in theories(args.theories)]
    missing = [n for n in names if not (args.out / f"{n}.npz").exists()]
    if missing:
        raise FileNotFoundError(f"no fit for {missing}: run `fit` for them first")
    if CHANCE not in names or RSA not in names:
        raise ValueError(f"the theories file must include {CHANCE} and {RSA}, the references of the table")
    lpd = {n: dict(np.load(args.out / f"{n}.npz")) for n in names}
    meta = {n: json.loads((args.out / f"{n}.json").read_text()) for n in names}
    table: Dict = {"blocks": [b.name for b in blocks], "theories": {}}
    for n in names:
        row = {"converged": meta[n]["converged"], "blocks": {}}
        for b in blocks:
            units = np.unique(np.asarray(b.units), return_inverse=True)[1]
            mine = lpd[n][b.name]
            cell = {"elpd": float(mine.sum())}
            for ref in (CHANCE, RSA):
                if n != ref:
                    other = lpd[ref][b.name]
                    cell[f"over_{ref}"] = float(mine.sum() - other.sum())
                    cell[f"se_{ref}"] = float(cluster_dse(mine, other, units))
            gain = np.bincount(units, weights=mine - lpd[CHANCE][b.name])
            cell["display_gain_over_chance"] = {"max": float(gain.max()), "min": float(gain.min()),
                                                "share_of_largest": float(gain.max() / max(gain.sum(), 1e-9))}
            row["blocks"][b.name] = cell
        table["theories"][n] = row
    (args.out / "cross_refit.json").write_text(json.dumps(table, indent=1))
    lines = ["| theory | " + " | ".join(table["blocks"]) + " |", "|---" * (len(blocks) + 1) + "|"]
    for n in names:
        if n == CHANCE:
            continue
        cells = []
        for b in table["blocks"]:
            c = table["theories"][n]["blocks"][b]
            cells.append(f"{c['over_chance']:.0f} ({c['over_chance'] / c['se_chance']:.1f})"
                         + ("" if n == RSA else f"; {c['over_rsa_l1']:+.0f} ({c['over_rsa_l1'] / c['se_rsa_l1']:+.1f})"))
        lines.append(f"| {n}{'' if table['theories'][n]['converged'] else ' (not converged)'} | " + " | ".join(cells) + " |")
    text = ("Each cell: out-of-fold elpd over chance (its clustered-SE ratio); then over plain RSA (rsa_l1) and its "
            "ratio.\n\n" + "\n".join(lines) + "\n")
    (args.out / "cross_refit.md").write_text(text)
    print(text)
    return table


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["fit", "report"])
    ap.add_argument("--open-corpus", type=Path, required=True)
    ap.add_argument("--fixed-corpus", type=Path, required=True)
    ap.add_argument("--theories", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=1000)
    args = ap.parse_args(argv)
    (fit if args.mode == "fit" else report)(args)


if __name__ == "__main__":
    main()
