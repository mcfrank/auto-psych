#!/usr/bin/env python3
"""Compare a raw-features (arm C) sweep against its featurized counterpart.

Both sweeps use the same ground truths and the same `BASE_SEED`, so repeat r is
the same synthetic dataset on both sides and the cells pair. This reports the
paired difference per ground truth, which is the evidence for or against making
`raw_features` the default: does removing the harness's feature columns cost
recovery, or was the featurizer propping up (or distorting) it?

Usage:
    uv run python scripts/subjective_randomness/compare_raw_features_arm.py \\
        --raw   $SCRATCH/auto-psych/holdout_raw_features \\
        --featurized $SCRATCH/auto-psych/recovery_improvement/recovery_2026_09_07/iter2/sweep
"""

from __future__ import annotations

import csv
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tyro
from pyprojroot import here

sys.path.insert(0, str(here()))


@dataclass
class Args:
    """Paired comparison of a raw-features sweep against a featurized one."""

    raw: Path
    """Run root of the raw-features sweep."""
    featurized: Path
    """Run root of the featurized sweep it pairs with."""
    out: Optional[Path] = None
    """Optional Markdown output (default: <raw>/COMPARISON.md)."""


def final_r_by_cell(root: Path) -> Dict[Tuple[str, str], Tuple[float, float, str]]:
    """``(run, gt) -> (seed-only r, final r, final best model)`` per finished cell."""
    out: Dict[Tuple[str, str], Tuple[float, float, str]] = {}
    for path in sorted(root.glob("run*/*/holdout.csv")):
        with path.open(newline="", encoding="utf-8") as fh:
            rows = sorted(
                csv.DictReader(fh), key=lambda r: float(r.get("global_step") or 0)
            )
        if not rows:
            continue
        key = (path.parent.parent.name, path.parent.name)
        out[key] = (
            float(rows[0]["pearson_r"]),
            float(rows[-1]["pearson_r"]),
            rows[-1].get("best_model", "?"),
        )
    return out


def render(raw_root: Path, feat_root: Path) -> str:
    raw, feat = final_r_by_cell(raw_root), final_r_by_cell(feat_root)
    gts = sorted({gt for _, gt in raw} | {gt for _, gt in feat})
    lines = [
        "# Raw features vs. the featurizer — paired comparison",
        "",
        f"- raw-features run: `{raw_root}` ({len(raw)} cells)",
        f"- featurized run:   `{feat_root}` ({len(feat)} cells)",
        "",
        "Same ground truths and the same `BASE_SEED`, so repeat *r* is the same",
        "synthetic dataset on both sides and only paired cells are differenced.",
        "",
        "| ground truth | paired cells | raw mean | featurized mean | mean paired diff | raw range |",
        "|---|---|---|---|---|---|",
    ]
    for gt in gts:
        pairs = [
            (raw[(run, gt)][1], feat[(run, gt)][1])
            for run, g in raw
            if g == gt and (run, gt) in feat
        ]
        if not pairs:
            lines.append(f"| {gt} | 0 | — | — | — | — |")
            continue
        r_vals = [p[0] for p in pairs]
        f_vals = [p[1] for p in pairs]
        diffs = [a - b for a, b in pairs]
        lines.append(
            f"| {gt} | {len(pairs)} | {statistics.fmean(r_vals):.3f} | "
            f"{statistics.fmean(f_vals):.3f} | {statistics.fmean(diffs):+.3f} | "
            f"{min(r_vals):.3f}–{max(r_vals):.3f} |"
        )
    lines += ["", "## Per cell", "", "| cell | raw seed-only | raw final | featurized final | diff | raw winner |", "|---|---|---|---|---|---|"]
    for (run, gt), (seed, fin, win) in sorted(raw.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        other = feat.get((run, gt))
        paired = f"{other[1]:.3f}" if other else "—"
        diff = f"{fin - other[1]:+.3f}" if other else "—"
        lines.append(
            f"| {run}/{gt} | {seed:.3f} | {fin:.3f} | {paired} | {diff} | {win} |"
        )
    missing = [f"{run}/{gt}" for (run, gt) in feat if (run, gt) not in raw]
    if missing:
        lines += ["", f"Unpaired (featurized only, {len(missing)}): " + ", ".join(sorted(missing)[:12])]
    lines += [
        "",
        "Read with care: a single cell per ground truth is not a result. In the",
        "featurized sweep `local_representativeness` ranged 0.406–0.991 across five",
        "repeats, so one high raw cell can be the same variance.",
        "",
    ]
    return "\n".join(lines)


def main(args: Args) -> None:
    out = args.out or (args.raw / "COMPARISON.md")
    text = render(args.raw, args.featurized)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
