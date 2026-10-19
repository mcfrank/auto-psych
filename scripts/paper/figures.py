"""The paper's Figures 2-4, one column (or row) per domain.

- Figure 2 (``fig2_elpd``): per domain, a column per run of ELPD
  differences from the best starting model (subjective randomness: the
  run's own seeds; reference games: the best promoted seed, the
  pre-registered bar): its starting models (dashes), the models it pruned
  (grey dots; the number game only), its other models live at the end (small
  dots) and its best discovered model (±2 SE of the paired difference,
  labelled). How each domain is scored
  is for the caption (the figure data's ``scoring``).
- Figure 3 (``fig3_conditions``): per domain, all runs' data pooled: each
  condition's observed choice proportion against the prediction of the best
  starting model (top row) and of the best discovered model (bottom row), both fitted
  on the pooled data (the figure data's ``pooled``), with the trial-weighted
  correlation and RMSE.
- Figure 4 (``fig4_pseudocode``): annotated pseudocode of each domain's
  best discovered model (Figure 3's), from
  ``scripts/paper/pseudocode/<domain>.yaml``. Each spec names its model's
  source and SHA-256; a spec whose source changed, or that describes another
  model than Figure 3's, raises.

Inputs are the JSON files written by ``sr_figure_data.py`` and
``rsa_figure_data.py`` (one per domain, ``<domain>.json``). A domain listed in
``--pending-domains`` is drawn as a labelled placeholder; any other missing
domain raises.

Usage:
    uv run python -m scripts.paper.figures
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tyro  # noqa: E402
import yaml  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from pyprojroot import here  # noqa: E402

DOMAINS = ["subjective_randomness", "rsa", "number_game"]
DOMAIN_TITLES = {
    "subjective_randomness": "Subjective randomness",
    "rsa": "Reference games",
    "number_game": "Number game",
}

# ColorBrewer Dark2, its first two colours (teal, orange); as a pair they
# pass the colour-vision-deficiency separation and contrast checks.
COLOR_DISCOVERED = "#1b9e77"
COLOR_SEED = "#d95f02"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"

# Figure 2's other models live at a run's end: a lighter shade of the
# discovered models' teal.
COLOR_OTHERS = "#48c9a0"
# Figure 2's models pruned during a run: a neutral grey.
COLOR_PRUNED = "#b9b7b0"
MOCK_MARK = " (MOCK DATA)"  # appended to a mock domain's titles

FULL_WIDTH_IN = 7.2  # a two-column paper's text width


@dataclass
class Args:
    figure_data: Path = here() / "data" / "paper" / "figure_data"
    """Directory with one ``<domain>.json`` per domain."""
    pseudocode: Path = here() / "scripts" / "paper" / "pseudocode"
    """Directory with one ``<domain>.yaml`` pseudocode spec per domain."""
    out_dir: Path = here() / "data" / "paper" / "figures"
    domains: List[str] = field(default_factory=lambda: list(DOMAINS))
    pending_domains: List[str] = field(default_factory=lambda: ["number_game"])
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


def run_label(domain: str, run: str) -> str:
    """A run's label. The RSA campaign's chains ``c0``, ``c1``, ... are its runs."""
    if domain == "rsa" and run.startswith("c") and run[1:].isdigit():
        return f"Run {int(run[1:]) + 1}"
    if run.startswith("run") and run[3:].isdigit():
        return f"Run {run[3:]}"
    return run


def panel_title(domain: str, *, mock_domains: List[str]) -> str:
    title = DOMAIN_TITLES[domain]
    return f"{title}{MOCK_MARK}" if domain in mock_domains else title


# --- statistics -------------------------------------------------------------


def weighted_fit_stats(observed: np.ndarray, predicted: np.ndarray, n: np.ndarray) -> Tuple[float, float]:
    """Trial-weighted Pearson correlation and RMSE between observed and
    predicted proportions. The correlation is NaN when either side is
    constant (a model predicting 0.5 everywhere has no correlation)."""
    w = np.asarray(n, dtype=float) / np.sum(n)
    o = np.asarray(observed, dtype=float)
    p = np.asarray(predicted, dtype=float)
    mo, mp = np.sum(w * o), np.sum(w * p)
    cov = np.sum(w * (o - mo) * (p - mp))
    var_product = np.sum(w * (o - mo) ** 2) * np.sum(w * (p - mp) ** 2)
    r = cov / np.sqrt(var_product) if var_product > 0 else float("nan")
    rmse = np.sqrt(np.sum(w * (o - p) ** 2))
    return float(r), float(rmse)


# --- drawing ----------------------------------------------------------------


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_SECONDARY)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK_SECONDARY, labelsize=7, width=0.6)


def _pending(ax, domain: str) -> None:
    ax.set_axis_off()
    ax.text(0.5, 0.5, f"{DOMAIN_TITLES[domain]}\nresults pending", ha="center", va="center",
            fontsize=8, color=INK_SECONDARY, transform=ax.transAxes)


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def elpd_columns(domain: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Figure 2's columns, one per run, each as differences from the best
    starting model: the run's starting models, the models it pruned (only a
    run whose figure data score them has ``pruned_during_run``), its other
    models live at the end and its winner with the SE of its paired
    difference."""
    columns = []
    for run in data["runs"]:
        seeds = [v["diff"] for name, v in run["vs_best_seed"].items() if name != run["best_model"]]
        columns.append({"label": run_label(domain, run["run"]), "seeds": seeds,
                        "pruned": list(run["pruned_during_run"].values()) if "pruned_during_run" in run else [],
                        "others": list(run["others_at_end"].values()),
                        "winner": run["vs_best_seed"][run["best_model"]]})
    return columns


def draw_elpd(domain_data: Dict[str, Optional[Dict[str, Any]]], out_dir: Path,
              mock_domains: List[str]) -> None:
    """Per domain, a column per run: its starting models (dashes), its other
    models live at the end (small dots) and its best discovered model (±2 SE,
    labelled with its difference), all from the best starting model."""
    fig, axes = plt.subplots(1, len(domain_data), figsize=(FULL_WIDTH_IN, 2.9))
    jitter = np.random.default_rng(0)
    for i, (ax, (domain, data)) in enumerate(zip(np.atleast_1d(axes), domain_data.items())):
        if data is None:
            _pending(ax, domain)
            continue
        _style(ax)
        ax.axhline(0.0, color=INK_SECONDARY, linewidth=0.8, zorder=0)
        columns = elpd_columns(domain, data)
        for x, column in enumerate(columns):
            for value in column["seeds"]:
                ax.plot([x - 0.40, x - 0.22], [value, value], color=COLOR_SEED, linewidth=1.1,
                        solid_capstyle="butt", zorder=2)
            pruned = np.asarray(column["pruned"], dtype=float)
            ax.scatter(x - 0.08 + jitter.uniform(-0.07, 0.07, pruned.size), pruned, s=8,
                       color=COLOR_PRUNED, linewidths=0, alpha=0.8, zorder=1)
            others = np.asarray(column["others"], dtype=float)
            ax.scatter(x + 0.09 + jitter.uniform(-0.05, 0.05, others.size), others, s=12,
                       color=COLOR_OTHERS, linewidths=0, alpha=0.8, zorder=2)
            win = column["winner"]
            ax.errorbar(x + 0.25, win["diff"], yerr=2 * win["se"], fmt="o", markersize=6,
                        color=COLOR_DISCOVERED, ecolor=COLOR_DISCOVERED, elinewidth=1.4,
                        capsize=0, markeredgecolor="white", markeredgewidth=0.8, zorder=3)
            ax.annotate(f"{win['diff']:+,.0f}", (x + 0.25, win["diff"] + 2 * win["se"]),
                        xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                        fontsize=6.5, color=INK)
        ax.set_xticks(range(len(columns)), [c["label"] for c in columns], fontsize=6.5)
        ax.set_xlim(-0.6, len(columns) - 0.4)
        ax.margins(y=0.12)
        if i == 0:
            ax.set_ylabel("ΔELPD vs. best starting model (nats)", fontsize=7, color=INK)
        ax.set_title(panel_title(domain, mock_domains=mock_domains), fontsize=8.5, color=INK,
                     loc="left")
    handles = [
        Line2D([], [], color=COLOR_SEED, linewidth=1.1, label="Starting models"),
        Line2D([], [], marker="o", color=COLOR_PRUNED, linestyle="none", markersize=3,
               label="Pruned during the run"),
        Line2D([], [], marker="o", color=COLOR_OTHERS, linestyle="none", markersize=3.5,
               label="Other models live at the end of the run"),
        Line2D([], [], marker="o", color=COLOR_DISCOVERED, linestyle="none", markersize=6,
               label="Best discovered model (±2 SE)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=6.5,
               bbox_to_anchor=(0.5, -0.01), handlelength=1.4, handletextpad=0.5,
               columnspacing=1.4)
    fig.tight_layout(w_pad=2.0, rect=(0, 0.07, 1, 1))
    _save(fig, out_dir, "fig2_elpd")


def draw_conditions(domain_data: Dict[str, Optional[Dict[str, Any]]], out_dir: Path,
                    mock_domains: List[str]) -> None:
    rows = [("best_seed", "Best starting model", COLOR_SEED),
            ("best_model", "Best discovered model", COLOR_DISCOVERED)]
    fig, axes = plt.subplots(2, len(domain_data), figsize=(FULL_WIDTH_IN, 4.9))
    axes = np.asarray(axes).reshape(2, len(domain_data))
    for col, (domain, data) in enumerate(domain_data.items()):
        for row, (key, label, color) in enumerate(rows):
            ax = axes[row, col]
            if data is None:
                _pending(ax, domain)
                continue
            _style(ax)
            conditions = data["conditions"]
            observed = np.array([c["observed"] for c in conditions])
            predicted = np.array([c[key] for c in conditions])
            n = np.array([c["n"] for c in conditions])
            ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.0, zorder=0)
            sizes = 6 + 30 * np.sqrt(n / n.max())
            ax.scatter(predicted, observed, s=sizes, marker="o", color=color, alpha=0.25,
                       linewidths=0, zorder=1)
            r, rmse = weighted_fit_stats(observed, predicted, n)
            r_text = "r undefined" if np.isnan(r) else f"r = {r:.2f}"
            ax.text(0.04, 0.96, f"{r_text}\nRMSE = {rmse:.3f}", transform=ax.transAxes,
                    fontsize=6.5, color=INK, va="top", ha="left")
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_aspect("equal")
            ax.set_xlabel(f"{label}: predicted", fontsize=7, color=INK)
            if col == 0:
                ax.set_ylabel("Observed proportion", fontsize=7, color=INK)
            if row == 0:
                ax.set_title(panel_title(domain, mock_domains=mock_domains),
                             fontsize=8.5, color=INK, loc="left")
    fig.tight_layout(h_pad=1.5, w_pad=1.5)
    _save(fig, out_dir, "fig3_conditions")


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


def main(args: Args) -> None:
    unknown = sorted(set(args.mock_domains) - set(args.domains))
    if unknown:
        raise ValueError(f"--mock-domains names domains that are not drawn: {unknown}")
    domain_data = load_domain_data(args)
    specs = {
        d: (None if data is None else load_pseudocode(args.pseudocode, d, data))
        for d, data in domain_data.items()
    }
    draw_elpd(domain_data, args.out_dir, args.mock_domains)
    draw_conditions(domain_data, args.out_dir, args.mock_domains)
    draw_pseudocode(specs, args.out_dir, args.mock_domains)
    print(f"wrote figures 2-4 to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
