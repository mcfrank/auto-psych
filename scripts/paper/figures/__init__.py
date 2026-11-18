"""The paper's Figures 2-4, one column (or row) per domain, from the figure data (one ``<domain>.json``
per domain, written by ``sr_figure_data.py``, ``rsa_figure_data.py`` and ``number_game_figure_data.py``).

- Figure 2 (``fig2.py``, ``fig2_elpd``): every model's ΔELPD against the best seed model, the runs
  collapsed. The number game's column is its pooled models scored on the Bigelow & Piantadosi responses
  no run drew (``number_game_unseen_items.py``; ``--ng-test-set``). How each domain is scored is for the
  caption (the figure data's ``scoring``).
- Figure 3 (``fig3.py``, ``fig3_conditions``) and Figure 4 (``fig4.py``, ``fig4_pseudocode``) keep the
  figure data's cross-validated comparison (and, until they are redrawn, the earlier style: ``legacy.py``).

The house style (``style.py``) is Avenir where installed. A domain listed in ``--pending-domains`` is drawn
as a labelled placeholder; any other missing domain raises.

Usage:
    uv run python -m scripts.paper.figures
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pyprojroot import here

from scripts.paper.figures import fig2, fig3, fig4
from scripts.paper.figures.data import forest_rows, number_game_unseen_view  # noqa: F401  (the tests' API)
from scripts.paper.figures.style import DOMAINS


@dataclass
class Args:
    figure_data: Path = here() / "data" / "paper" / "figure_data"
    """Directory with one ``<domain>.json`` per domain."""
    pseudocode: Path = here() / "scripts" / "paper" / "pseudocode"
    """Directory with one ``<domain>.yaml`` pseudocode spec per domain."""
    model_names: Path = here() / "scripts" / "paper" / "model_names.yaml"
    """Human-readable names of the models Figure 2 labels, per domain."""
    seed_papers: Path = here() / "scripts" / "paper" / "seed_papers.yaml"
    """The paper each starting model implements (or null), per domain."""
    out_dir: Path = here() / "data" / "paper" / "figures"
    ng_unseen: Path = here() / "data" / "paper" / "number_game_unseen_items.json"
    """The number game's pooled models scored on unseen items: Figure 2's column."""
    ng_test_set: Literal["unseen_items", "unseen_example_sets"] = "unseen_items"
    """Which of its test sets Figure 2 draws: the (set, target) items no run drew, or the stricter
    example sets no run drew (11 sets, too few to separate the models)."""
    figures: List[Literal["2", "3", "4"]] = field(default_factory=lambda: ["2", "3", "4"])
    """Which figures to draw."""
    domains: List[str] = field(default_factory=lambda: list(DOMAINS))
    pending_domains: List[str] = field(default_factory=list)
    """Domains drawn as a 'pending' placeholder (their data are not in yet)."""
    mock_domains: List[str] = field(default_factory=list)
    """Domains whose figure data are mock (for drafting the figures' look):
    their panel titles say so."""


# --- inputs -----------------------------------------------------------------


def load_domain_data(args: Args) -> Dict[str, Optional[Dict[str, Any]]]:
    """Each domain's figure data, ``None`` for a pending domain."""
    loaded: Dict[str, Optional[Dict[str, Any]]] = {}
    for domain in args.domains:
        if domain in args.pending_domains:
            loaded[domain] = None
            continue
        path = args.figure_data / f"{domain}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"no figure data for {domain!r} at {path} (pass it in --pending-domains "
                "to draw a placeholder)"
            )
        data = json.loads(path.read_text())
        if data["domain"] != domain:
            raise RuntimeError(f"{path} holds domain {data['domain']!r}, not {domain!r}")
        loaded[domain] = data
    return loaded


def main(args: Args) -> None:
    unknown = sorted(set(args.mock_domains) - set(args.domains))
    if unknown:
        raise ValueError(f"--mock-domains names domains that are not drawn: {unknown}")
    domain_data = load_domain_data(args)
    if "4" in args.figures:  # its specs checked before anything is drawn
        specs = {d: (None if data is None else fig4.load_pseudocode(args.pseudocode, d, data))
                 for d, data in domain_data.items()}
    if "2" in args.figures:
        names = yaml.safe_load(args.model_names.read_text())
        papers = yaml.safe_load(args.seed_papers.read_text())
        drawn = [d for d, data in domain_data.items() if data is not None]
        fig2_data = dict(domain_data)
        if fig2_data.get("number_game") is not None:
            fig2_data["number_game"] = number_game_unseen_view(
                domain_data["number_game"], json.loads(args.ng_unseen.read_text()), args.ng_test_set)
        fig2.draw(fig2_data, args.out_dir, args.mock_domains, {d: names[d] for d in drawn},
                  {d: papers[d] for d in drawn})
    if "3" in args.figures:
        fig3.draw_conditions(domain_data, args.out_dir, args.mock_domains)
    if "4" in args.figures:
        fig4.draw_pseudocode(specs, args.out_dir, args.mock_domains)
    print(f"wrote figure(s) {', '.join(args.figures)} to {args.out_dir}", flush=True)
