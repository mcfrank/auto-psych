"""The paper's Figures 1-4. Figure 1 is drawn by hand (``fig1.html``); Figures 2-4, one column (or row) per
domain, from the figure data (one ``<domain>.json`` per domain, written by ``sr_figure_data.py``,
``rsa_figure_data.py`` and ``number_game_figure_data.py``).

- Figure 1 (``fig1.py``, ``fig1_loops``): the framework's two loops and the three domains, printed from
  ``fig1.html`` by Playwright's Chromium (``uv run playwright install chromium`` once). Not in the default
  set: draw it with ``--figures 1``, where Avenir and Menlo are installed (it refuses to substitute fonts).
- Figure 2 (``fig2.py``, ``fig2_elpd``): every model's ΔELPD against the best seed model, the runs
  collapsed. The number game's column is its pooled models scored on the Bigelow & Piantadosi responses
  no run drew (``number_game_unseen_items.py``; ``--ng-test-set``). How each domain is scored is for the
  caption (the figure data's ``scoring``).
- Figure 3 (``fig3.py``, ``fig3_effects``): overall fit and a classic effect on Figure 2's split
  (``<domain>_heldout_conditions.json``), and a new one from the figure data's in-sample conditions.
- Figure 4 (``fig4.py``, ``fig4_ablation``): each winner's backward elimination, its mechanisms named in
  ``mechanisms.yaml``.

The house style (``style.py``) is Avenir where installed. A domain listed in ``--pending-domains`` is drawn
as a labelled placeholder; any other missing domain raises.

Usage:
    uv run python -m scripts.paper.figures               # Figures 2-4
    uv run python -m scripts.paper.figures --figures 1   # Figure 1
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import matplotlib.pyplot as plt
import yaml
from pyprojroot import here

from scripts.paper.figures import fig1, fig2, fig3, fig4
from scripts.paper.figures.data import forest_rows, number_game_unseen_view  # noqa: F401  (the tests' API)
from scripts.paper.figures.style import DOMAINS, setup_style


@dataclass
class Args:
    figure_data: Path = here() / "data" / "paper" / "figure_data"
    """Directory with one ``<domain>.json`` per domain."""
    rsa_displays: Path = here() / "data" / "rsa" / "viz" / "panel_displays.json"
    """What each reference-game display showed (``scripts/paper/rsa_panel_displays.py``): Figure 3's games."""
    mechanisms: Path = here() / "scripts" / "paper" / "mechanisms.yaml"
    """Figure 4's name and origin of each winner's mechanisms, per domain."""
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
    histories: Dict[str, str] = field(default_factory=lambda: {
        "subjective_randomness": str(here() / "data/results/human_experiment_sr-oct26/collected/{run}/subjective_randomness"),
        "rsa": str(here() / "data/rsa/live/live_rsa_{run}/outputs"),
        "number_game": str(here() / "data/results/number_game_dataset-nov26/collected/{run}/number_game")})
    """Each domain's run directory (``{run}``: the run), whose experiments' histories Figure 2 traces."""
    figures: List[Literal["1", "2", "3", "4"]] = field(default_factory=lambda: ["2", "3", "4"])
    """Which figures to draw. Figure 1 is drawn only when named (``--figures 1``): it needs Playwright's
    Chromium and the Avenir and Menlo fonts (macOS)."""
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
    if "1" in args.figures:
        fig1.draw(args.out_dir)
    if "2" in args.figures:
        names = yaml.safe_load(args.model_names.read_text())
        papers = yaml.safe_load(args.seed_papers.read_text())
        drawn = [d for d, data in domain_data.items() if data is not None]
        fig2_data = dict(domain_data)
        if fig2_data.get("number_game") is not None:
            fig2_data["number_game"] = number_game_unseen_view(
                domain_data["number_game"], json.loads(args.ng_unseen.read_text()), args.ng_test_set)
        fig2.draw(fig2_data, args.out_dir, args.mock_domains, {d: names[d] for d in drawn},
                  {d: papers[d] for d in drawn}, args.histories)
    with plt.rc_context():
        setup_style()
        if "3" in args.figures:
            fig3.draw(args.figure_data, args.rsa_displays, args.out_dir)
        if "4" in args.figures:
            fig4.draw_ablation(args.figure_data, args.mechanisms, args.out_dir)
    print(f"wrote figure(s) {', '.join(args.figures)} to {args.out_dir}", flush=True)
