"""Figure 4 (``fig4_pseudocode``): annotated pseudocode of each domain's best discovered model
(Figure 3's), from ``scripts/paper/pseudocode/<domain>.yaml``. Each spec names its model's source and
SHA-256; a spec whose source changed, or that describes another model than Figure 3's, raises."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import yaml
from pyprojroot import here

from scripts.paper.figures.legacy import COLOR_DISCOVERED, FULL_WIDTH_IN, INK, INK_SECONDARY, _pending, _save
from scripts.paper.figures.style import MOCK_MARK

def load_pseudocode(spec_dir: Path, domain: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """The domain's pseudocode spec, checked against its model's source and
    against the best model on the pooled data (Figure 3's)."""
    spec = yaml.safe_load((spec_dir / f"{domain}.yaml").read_text())
    source = Path(spec["source"])
    if not source.is_absolute():
        source = here() / source
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if sha != spec["sha256"]:
        raise RuntimeError(
            f"{domain}: the pseudocode's sha256 does not match {source} (the model "
            "changed; update the pseudocode, then its sha256)"
        )
    best_model = data["pooled"]["best_model"]
    if spec["model"] != best_model:
        raise RuntimeError(
            f"{domain}: the pseudocode describes {spec['model']!r}, not the best model on "
            f"the pooled data, {best_model!r}"
        )
    return spec



def draw_pseudocode(specs: Dict[str, Optional[Dict[str, Any]]], out_dir: Path,
                    mock_domains: List[str]) -> None:
    line_in = 0.16
    heights = [(len(s["lines"]) + 2.2) * line_in if s else 0.6 for s in specs.values()]
    fig, axes = plt.subplots(len(specs), 1, figsize=(FULL_WIDTH_IN, sum(heights)),
                             gridspec_kw={"height_ratios": heights})
    for ax, (domain, spec), height in zip(np.atleast_1d(axes), specs.items(), heights):
        if spec is None:
            _pending(ax, domain)
            continue
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, height)
        top = height - 0.5 * line_in
        title = f"{spec['title']}{MOCK_MARK}" if domain in mock_domains else spec["title"]
        ax.text(0.0, top, title, fontsize=8.5, color=INK, va="top", fontweight="bold")
        ax.text(0.0, top - 1.1 * line_in, spec["model"], fontsize=6.5, color=INK_SECONDARY,
                va="top", family="monospace")
        for i, line in enumerate(spec["lines"]):
            y = top - (i + 2.2) * line_in
            ax.text(0.01, y, line["code"], fontsize=7, color=INK, va="top", family="monospace")
            if line.get("note"):
                ax.text(0.64, y, f"← {line['note']}", fontsize=6.5, color=COLOR_DISCOVERED,
                        va="top", style="italic")
    _save(fig, out_dir, "fig4_pseudocode")
