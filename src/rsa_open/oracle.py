"""How close a rehearsal's loop came to its hidden ground truth (PI 2026-10-26).

    python -m src.rsa_open.oracle --loop-dir <run>/experiment<n>/loop --truth-dir <private>/ground_truth \
        --out <private>/oracle

A rehearsal's experiments are simulated from a hidden truth (`src.rsa_open.ground_truth`), so the
truth's own theory, fitted like any other, is the yardstick for the loop's theories on those
experiments: the best held-out score a theory could expect there. This scores it **on the loop's
own corpus and folds**, beside every theory the loop ended with and every theory it pruned, with
the loop's settings. The loop's fits are in its cache (``--loop-dir``'s ``.fit_cache``, linked,
not copied), so only the truth is fitted anew.

Out: ``oracle.json`` and ``oracle.md``: per block, each theory's ELPD-CV and its gap to the truth's
with the unit-clustered SE. E is real people, not the truth's, so on E the truth is one more
theory, not a ceiling. The truth appears as ``ground_truth``, never by its name. Harness only: no
agent sees this module or its output (`scripts/rsa/slurm/agent_tree.exclude`).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import tyro
import yaml

TRUTH_NAME = "ground_truth"


@dataclass
class Args:
    loop_dir: Path
    """The rehearsal loop to score (its corpus.json, models/ and .fit_cache)."""
    truth_dir: Path
    """The private ground-truth directory (ground_truth.json and its theory file)."""
    out: Path
    """A new private directory."""
    cv_folds: int = 5


def _models_dir(loop_dir: Path, truth_file: Path, dst: Path) -> List[str]:
    """Every theory the loop ended with or pruned, and the truth, as one seed directory."""
    dst.mkdir(parents=True)
    entries = []
    for src in (loop_dir / "models", loop_dir / "models" / "pruned"):
        manifest = src / "models_manifest.yaml"
        rationales = {e["name"]: e.get("rationale", "") for e in yaml.safe_load(manifest.read_text())["models"]} \
            if manifest.exists() else {}
        for f in sorted(src.glob("*.py")):
            shutil.copyfile(f, dst / f.name)  # the same bytes: the loop's cached fits are hits
            entries.append(dict(name=f.stem, rationale=rationales.get(f.stem, "")))
    if any(e["name"] == TRUTH_NAME for e in entries):
        raise ValueError(f"the loop has a theory named {TRUTH_NAME}")
    shutil.copyfile(truth_file, dst / f"{TRUTH_NAME}.py")
    entries.append(dict(name=TRUTH_NAME, rationale="the hidden ground truth"))
    (dst / "models_manifest.yaml").write_text(yaml.safe_dump({"models": entries}, sort_keys=False))
    return [e["name"] for e in entries]


def score(args: Args, **loop_kw) -> Dict:
    """``loop_kw``: `OpenLoopConfig` overrides (tests: small settings); the default is the loop's."""
    from src.models.clustered_se import cluster_dse
    from src.rsa_open.ground_truth import load_ground_truth
    from src.rsa_open.loop.orchestrator import OpenLoop, OpenLoopConfig

    loop_dir, out = Path(args.loop_dir), Path(args.out)
    if out.exists():
        raise FileExistsError(f"{out} exists")
    truth = load_ground_truth(args.truth_dir)
    names = _models_dir(loop_dir, Path(truth.theory_path), out / "seeds")
    results = out / "loop"
    results.mkdir(parents=True)
    (results / ".fit_cache").symlink_to((loop_dir / ".fit_cache").resolve())

    def no_agents(*_):
        raise RuntimeError("the oracle runs no agents")

    loop = OpenLoop(OpenLoopConfig(corpus_path=loop_dir / "corpus.json", seed_models_dir=out / "seeds",
                                   results_dir=results, cv_folds=args.cv_folds, critique=False, **loop_kw), no_agents)
    loop.setup()
    missing = sorted(set(names) - set(loop.live))
    s = loop.standing()
    blocks = {b.name: b for b in loop.blocks}
    table: Dict[str, Dict] = {}
    for name in loop.live:
        cv = loop.cv.get(name)
        row = {"trusted": s[name]["trusted"], "by_block": {}}
        for b, block in blocks.items():
            if cv is None or loop.cv.get(TRUTH_NAME) is None:
                continue
            units = np.unique(np.asarray(block.units), return_inverse=True)[1]
            mine, theirs = cv.pointwise[b], loop.cv[TRUTH_NAME].pointwise[b]
            row["by_block"][b] = dict(elpd_cv=float(mine.sum()), minus_truth=float(mine.sum() - theirs.sum()),
                                      dse=float(cluster_dse(theirs, mine, units)))
        table[name] = row
    record = dict(loop_dir=str(loop_dir), blocks=list(blocks), dropped=missing, theories=table,
                  ended_live=sorted(p.stem for p in (loop_dir / "models").glob("*.py")))
    (out / "oracle.json").write_text(json.dumps(record, indent=1))
    (out / "oracle.md").write_text(markdown(record))
    return record


def markdown(record: Dict) -> str:
    blocks = record["blocks"]
    lines = ["# The loop's theories against the hidden truth", "",
             "ELPD-CV on the loop's corpus and folds; \"− truth\" is the theory's minus the truth's "
             "(negative: worse), ± the unit-clustered SE. E is real people; the later blocks are the truth's.", "",
             "| theory | ended live | " + " | ".join(f"{b} elpd_cv | {b} − truth" for b in blocks) + " |",
             "|---|---|" + "---|---|" * len(blocks)]
    last = blocks[-1]
    order = sorted(record["theories"], key=lambda n: -record["theories"][n]["by_block"].get(last, {}).get("elpd_cv",
                                                                                                        -np.inf))
    for n in order:
        r = record["theories"][n]
        cells = []
        for b in blocks:
            d = r["by_block"].get(b)
            cells += ["—", "—"] if d is None else [f"{d['elpd_cv']:.1f}", f"{d['minus_truth']:+.1f} ± {d['dse']:.1f}"]
        live = "truth" if n == TRUTH_NAME else ("yes" if n in record["ended_live"] else "pruned")
        lines.append(f"| {n}{'' if r['trusted'] else ' (untrusted)'} | {live} | " + " | ".join(cells) + " |")
    if record["dropped"]:
        lines += ["", f"Not fitted (a failure of their own): {', '.join(record['dropped'])}."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    from src.rsa.cpus import pin_main_thread

    pin_main_thread()
    print(markdown(score(tyro.cli(Args))))
