"""The paper's Figures 2-4, one column (or row) per domain.

- Figure 2 (``fig2_elpd``, models across and ELPD up; ``fig2_elpd_horizontal``,
  the same as a forest plot): per domain, each model's ELPD, with the best
  starting model's dashed, the runs collapsed (subjective randomness:
  all runs pooled; reference games: every chain on the pre-registered test,
  against the best promoted seed; the number game's one run): the best
  discovered models and the starting models as named points with ±2 SE of
  the paired difference (names from ``model_names.yaml``, the starting
  models' papers from ``seed_papers.yaml``), and the non-winning models
  (the other models live at the end and those pruned during the runs) as
  one strip in admission order. How each
  domain is scored is for the caption (the figure data's ``scoring``).
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
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402
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

# Figure 2's models the loop wrote, coloured by the run that wrote them
# (run 1 the discovered models' teal). With the starting models' orange the
# four pass the palette validator's checks on every pair (worst colour-
# vision-deficiency separation ΔE 9.3, deutan; normal vision 20.7).
RUN_COLORS = ["#1b9e77", "#3b5bdb", "#b5338a"]
# A run's non-winning models: its colour, part greyed (mixed with a neutral grey).
NEUTRAL_GREY = "#b9b7b0"
NONWINNING_GREY_SHARE = 0.35
NONWINNING_COLORS = [tuple((1 - NONWINNING_GREY_SHARE) * np.asarray(matplotlib.colors.to_rgb(c))
                           + NONWINNING_GREY_SHARE * np.asarray(matplotlib.colors.to_rgb(NEUTRAL_GREY)))
                     for c in RUN_COLORS]
MOCK_MARK = " (MOCK DATA)"  # appended to a mock domain's titles

FULL_WIDTH_IN = 7.2  # a two-column paper's text width


@dataclass
class Args:
    figure_data: Path = here() / "data" / "paper" / "figure_data"
    """Directory with one ``<domain>.json`` per domain."""
    pseudocode: Path = here() / "scripts" / "paper" / "pseudocode"
    model_names: Path = here() / "scripts" / "paper" / "model_names.yaml"
    """Human-readable names of the models Figure 2 labels, per domain."""
    seed_papers: Path = here() / "scripts" / "paper" / "seed_papers.yaml"
    """The paper each starting model implements (or null), per domain."""
    """Directory with one ``<domain>.yaml`` pseudocode spec per domain."""
    out_dir: Path = here() / "data" / "paper" / "figures"
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


# Figure 2 collapses each domain's runs into one comparison. Subjective
# randomness scores each run on its own data, so it is drawn from all runs
# pooled (the figure data's ``pooled``); the other domains score every run on
# one test (reference games: the pre-registered experiment; the number game
# has one run), so their runs are drawn together.
POOLED_VIEW = {"subjective_randomness"}
# The pooled view's labels of the runs' models, "<run>__<name>"
# (sr_figure_data.POOLED_LABEL_SEP).
POOLED_LABEL_SEP = "__"


def best_seed_elpd(domain: str, data: Dict[str, Any]) -> float:
    """The best starting model's own ELPD: Figure 2's differences are drawn
    shifted by it (raw ELPD). Every run on one test must agree."""
    if domain in POOLED_VIEW:
        return float(data["pooled"]["elpd"][data["pooled"]["best_seed"]])
    found = {round(float(r["elpd"][r["best_seed"]]), 6) for r in data["runs"] if r["best_seed"] in r["elpd"]}
    if len(found) != 1:
        raise RuntimeError(f"{domain}: the runs record {sorted(found)} as the best starting model's ELPD")
    return found.pop()


def _model_name(names: Dict[str, str], domain: str, model: str) -> str:
    if model not in names:
        raise KeyError(f"{domain}: {model!r} has no name in model_names.yaml")
    return names[model]


def _nonwinning_strip(domain: str, by_run: Dict[str, Dict[str, float]],
                      order: Dict[str, int]) -> Dict[str, Any]:
    """The non-winning models (pruned during a run or live at its end) as one
    strip, run by run in admission order: their ``values``, ``progress``
    through their own run (rank in admission order, scaled to 0..1) and
    ``runs`` (each run's index in ``by_run``)."""
    strip: Dict[str, List[Any]] = {"values": [], "progress": [], "runs": []}
    for k, (run, models) in enumerate(by_run.items()):
        missing = sorted(set(models) - set(order))
        if missing:
            raise RuntimeError(f"{domain} {run}: no admission order for {missing}")
        ranked = sorted(models, key=order.get)
        for i, label in enumerate(ranked):
            strip["values"].append(models[label])
            strip["progress"].append(i / (len(ranked) - 1) if len(ranked) > 1 else 0.5)
            strip["runs"].append(k)
    return strip


def _seed_paper(papers: Dict[str, Optional[str]], domain: str, model: str) -> Optional[str]:
    if model not in papers:
        raise KeyError(f"{domain}: starting model {model!r} is not in seed_papers.yaml "
                       f"(list it, with null if no paper proposed it)")
    return papers[model]


def forest_rows(domain: str, data: Dict[str, Any], names: Dict[str, str],
                papers: Dict[str, Optional[str]]) -> List[Dict[str, Any]]:
    """Figure 2's rows for one domain, top to bottom: the best discovered
    models (named; their run is their colour), the non-winning models (the
    other models live at the end and those pruned during the runs: one
    unnamed strip), then the starting models (named, each with the
    ``paper`` it implements or ``None``), each against the best starting
    model.

    A labelled row has ``diff`` and ``se`` (``None``: its paired SE is not
    computed yet; drawn hollow, without a CI). The strip has ``values``, or
    ``None`` when its models are not scored yet (drawn as "pending"), and
    each model's ``progress`` through its own run, 0 for the first it
    admitted to 1 for the last (``pruned_order`` and ``others_order``,
    ranked within each run, so collapsed runs overlay) and its ``runs``. A
    best discovered model has its ``run`` (each
    run's index in the domain's run order: Figure 2 colours by run)."""
    if domain in POOLED_VIEW:
        pooled = data["pooled"]
        best_seed = pooled["best_seed"]
        paired = pooled.get("vs_best_seed")
        top = pooled["elpd"][best_seed]
        winners = [(m, run) for m, run in pooled["winner_runs"].items()]
        seeds = [m for m in pooled["elpd"] if m not in pooled["winner_runs"]]

        def value(m: str) -> Dict[str, Any]:
            if paired is not None:
                return {"diff": paired[m]["diff"], "se": paired[m]["se"]}
            return {"diff": pooled["elpd"][m] - top, "se": 0.0 if m == best_seed else None}

        scored = "pruned_during_run" in pooled
        run_names = list(pooled["runs"]) if scored else sorted(set(pooled["winner_runs"].values()))
        strip = None
        if scored:
            every = {**pooled["pruned_during_run"], **pooled["others_at_end"]}
            by_run = {run: {label: v for label, v in every.items()
                            if label.startswith(f"{run}{POOLED_LABEL_SEP}")} for run in run_names}
            strip = _nonwinning_strip(domain, by_run,
                                      {**pooled["pruned_order"], **pooled["others_order"]})
    else:
        runs = data["runs"]
        if len({r["best_seed"] for r in runs}) != 1:
            raise RuntimeError(f"{domain}: the runs are scored against different best starting models")
        winners = [(r["best_model"], r["run"]) for r in runs]
        seed_values: Dict[str, Dict[str, Any]] = {}
        for r in runs:
            for m, v in r["vs_best_seed"].items():
                if m == r["best_model"]:
                    continue
                if m in seed_values and abs(seed_values[m]["diff"] - v["diff"]) > 1e-6:
                    raise RuntimeError(f"{domain}: {m!r} differs between runs on one test")
                seed_values[m] = {"diff": v["diff"], "se": v["se"]}
        seeds = list(seed_values)
        by_winner = {r["best_model"]: r["vs_best_seed"][r["best_model"]] for r in runs}

        def value(m: str) -> Dict[str, Any]:
            v = by_winner.get(m, seed_values.get(m))
            return {"diff": v["diff"], "se": v["se"]}

        has_pruned = {"pruned_during_run" in r for r in runs}
        if len(has_pruned) != 1:
            raise RuntimeError(f"{domain}: some runs score their pruned models and some do not")
        scored = has_pruned.pop()
        run_names = [r["run"] for r in runs]
        strip = None
        if scored:
            sep = POOLED_LABEL_SEP
            order = {f"{r['run']}{sep}{n}": i for r in runs
                     for n, i in {**r["pruned_order"], **r["others_order"]}.items()}
            by_run = {r["run"]: {f"{r['run']}{sep}{n}": v
                                 for n, v in {**r["pruned_during_run"], **r["others_at_end"]}.items()}
                      for r in runs}
            strip = _nonwinning_strip(domain, by_run, order)

    rows = [{"kind": "winner", "model": m, "label": _model_name(names, domain, m),
             "run": run_names.index(run), **value(m)} for m, run in winners]
    rows.sort(key=lambda r: -r["diff"])
    rows.append({"kind": "nonwinning", "label": "Non-winning models",
                 **(strip or {"values": None})})
    seed_rows = [{"kind": "seed", "model": m, "label": _model_name(names, domain, m),
                  "paper": _seed_paper(papers, domain, m), **value(m)} for m in seeds]
    return rows + sorted(seed_rows, key=lambda r: -r["diff"])


NONWINNING_SPAN = 3.5  # the non-winning strip's width in columns: its dots spread by progress
FIG2_HEIGHT_IN = 2.3


SEED_WIDTH = 0.4  # a starting model's column width (the others' is 1)


def _column_width(column: Dict[str, Any]) -> float:
    return {"nonwinning": NONWINNING_SPAN, "seed": SEED_WIDTH}.get(column["kind"], 1.0)


def _column_positions(columns: List[Dict[str, Any]]) -> List[float]:
    """Each column's centre, left to right (``_column_width`` each)."""
    centres, edge = [], 0.0
    for column in columns:
        width = _column_width(column)
        centres.append(edge + width / 2)
        edge += width
    return centres


def _strip_label(row: Dict[str, Any]) -> str:
    return row["label"] if row["values"] is None else f"{row['label']} ({len(row['values'])})"


def _seed_names(columns: List[Dict[str, Any]]) -> Dict[str, str]:
    """The vertical Figure 2's name for each starting model it names: those
    a paper proposed, by the paper ("Kahneman & Tversky (1972)"; one name
    for the models of a paper that proposed several), and the best starting
    model, by its name where no paper did."""
    seeds = [c for c in columns if c["kind"] == "seed"]
    best = max(seeds, key=lambda c: c["diff"])
    named = {}
    for c in seeds:
        if c["paper"] is not None:
            authors, year = c["paper"].rsplit(", ", 1)
            named[c["model"]] = f"{authors} ({year})"
        elif c is best:
            named[c["model"]] = c["label"]
    return named


NAME_LINE_PT = 7.0  # one line of a band of names


def _name_band(ax, fig, items: List[Tuple[str, List[Tuple[float, float]]]], at: str) -> None:
    """Name the (text, points) in a band of one line each (a leader line to
    each point; the first point places the name), at the top of
    the panel (``at`` "top": the best discovered models, highest on top, each
    name ending at its point's column) or at the bottom ("bottom": the
    starting models, lowest at the bottom, each name starting at its point's
    column), shifted right where a name would cross the y-axis (and a bottom
    name left where it would run past the panel's right edge), with a leader
    line to its point. The panel's range grows to hold the band, whose ticks
    are hidden."""
    if not items:
        return
    top = at == "top"
    height_pt = ax.bbox.height * 72.0 / fig.dpi
    share = (len(items) * NAME_LINE_PT + 4.0) / height_pt
    lo, hi = ax.get_ylim()
    ax.set_yticks([v for v in ax.get_yticks() if lo <= v <= hi])
    if top:
        hi = (hi - lo * share) / (1 - share)
    else:
        lo = (lo - hi * share) / (1 - share)
    ax.set_ylim(lo, hi)
    per_pt = (hi - lo) / height_pt
    renderer = fig.canvas.get_renderer()
    frame = ax.get_window_extent(renderer)
    to_data = ax.transData.inverted()
    ha = "right" if top else "left"
    for k, (text, points) in enumerate(sorted(items, key=lambda w: -w[1][0][1] if top else w[1][0][1])):
        x, y = points[0]
        y_text = hi - (k + 0.5) * NAME_LINE_PT * per_pt if top else lo + (k + 0.5) * NAME_LINE_PT * per_pt
        probe = ax.text(x, y_text, text, ha=ha, va="center", fontsize=5)
        box = probe.get_window_extent(renderer)
        probe.remove()
        x_text = x if box.x0 >= frame.x0 else to_data.transform((box.x1 + frame.x0 - box.x0, 0))[0]
        if not top and box.x0 < frame.x0:
            x_text = to_data.transform((frame.x0, 0))[0]
        if not top and box.x1 > frame.x1:
            x_text = to_data.transform((frame.x1 - box.width, 0))[0]
        name = ax.annotate(text, (x, y), xytext=(x_text, y_text), textcoords="data", ha=ha,
                           va="center", fontsize=5, color=INK, annotation_clip=False, zorder=4,
                           arrowprops=dict(arrowstyle="-", color=INK_SECONDARY, linewidth=0.4,
                                           shrinkA=0, shrinkB=2.5,
                                           relpos=(1.0, 0.5) if top else (0.0, 0.5)))
        name.set_in_layout(False)
        for more in points[1:]:
            leader = ax.annotate("", more, xytext=(x_text, y_text), textcoords="data",
                                 annotation_clip=False, zorder=4,
                                 arrowprops=dict(arrowstyle="-", color=INK_SECONDARY,
                                                 linewidth=0.4, shrinkA=0, shrinkB=2.5))
            leader.set_in_layout(False)


def _elpd_ticks(axis) -> None:
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}".replace("-", "−")))


LEGEND_ROW_PT = 10.0  # one row of Figure 2's legend


def _fig2_legend(fig, hollow: List[str], n_runs: int) -> float:
    """Figure 2's legend, at the top: the kinds of model (the loop's in
    neutral ink), then the runs' colours. Returns the figure fraction it
    takes."""
    def dot(color, label, size=3.6, **kw):
        return Line2D([], [], marker="o", color=color, linestyle="none", markersize=size,
                      label=label, **kw)

    kinds = [
        dot(COLOR_SEED, "Starting models"),
        Line2D([], [], color=INK_SECONDARY, linewidth=0.7, linestyle=(0, (3, 2)),
               label="Best starting model"),
        dot(NEUTRAL_GREY, "Non-winning models (in the order tried)", size=2.8),
        Line2D([], [], marker="o", color=INK_SECONDARY, markersize=3.6, linestyle="none",
               label="Best discovered (±2 SE)"),
    ]
    if hollow:
        print(f"[fig2] drawn without a CI (paired SE not computed yet): {hollow}", flush=True)
        kinds.append(dot(INK_SECONDARY, "SE not yet computed", markerfacecolor="white",
                         markeredgewidth=0.9))
    runs = [Line2D([], [], marker="s", color=RUN_COLORS[k], linestyle="none", markersize=3.6,
                   label=f"Run {k + 1}") for k in range(n_runs)]
    row = LEGEND_ROW_PT / 72.0 / fig.get_figheight()
    for k, handles in enumerate((kinds, runs)):
        legend = fig.legend(handles=handles, loc="upper left", ncol=len(handles), frameon=False,
                            fontsize=5.5, bbox_to_anchor=(0.0, 1.0 - k * row), handlelength=1.4,
                            handletextpad=0.4, columnspacing=1.0)
        if k == 0:
            fig.add_artist(legend)  # the second call would replace it
    return 2 * row + 0.01


def _strip(ax, xs, ys, runs: List[int]) -> None:
    """The non-winning models, each in its run's part-greyed colour, translucent."""
    ax.scatter(xs, ys, s=6, c=[NONWINNING_COLORS[r] for r in runs], linewidths=0, alpha=0.55,
               zorder=2)


def _n_runs(rows_by_domain: Dict[str, Optional[List[Dict[str, Any]]]]) -> int:
    return max(r["run"] + 1 for rows in rows_by_domain.values() if rows is not None
               for r in rows if r["kind"] == "winner")


def _point(ax, at: Tuple[float, float], half: float, color: str, se_known: bool, vertical: bool) -> None:
    """A starting or best discovered model: a point with ±``half`` (2 SE),
    hollow when its SE is not computed yet."""
    if not se_known:
        ax.plot(*at, "o", markersize=3.6, markerfacecolor="white", markeredgecolor=color,
                markeredgewidth=0.9, zorder=3)
        return
    ax.errorbar(*at, **{"yerr" if vertical else "xerr": half}, fmt="o", markersize=3.6,
                color=color, ecolor=color, elinewidth=1.0, capsize=0, markeredgecolor="white",
                markeredgewidth=0.4, zorder=3)


def draw_elpd(domain_data: Dict[str, Optional[Dict[str, Any]]], out_dir: Path,
              mock_domains: List[str], names: Dict[str, Dict[str, str]],
              papers: Dict[str, Dict[str, Optional[str]]]) -> None:
    """Per domain, side by side, each model's ELPD (up is a better fit; the
    differences from the best starting model shifted by its own ELPD, which a
    dashed line marks), left to right: the starting models (worst first),
    the models pruned during the run (left to right in the order each run
    admitted them), the other live models, and the best discovered models;
    starting and best discovered models as points with ±2 SE. The best
    discovered models are named in a band at the top, the starting models a
    paper proposed (by the paper) and the best starting model in a band at
    the bottom. Thin rules separate the groups."""
    rows_by_domain = {d: None if data is None else forest_rows(d, data, names[d], papers[d])[::-1]
                      for d, data in domain_data.items()}
    widths = [6.0 if cols is None else _column_positions(cols)[-1] + 0.5
              for cols in rows_by_domain.values()]
    fig, axes = plt.subplots(1, len(widths), figsize=(FULL_WIDTH_IN, FIG2_HEIGHT_IN),
                             gridspec_kw={"width_ratios": widths})
    hollow: List[str] = []
    drawn = []
    bands: Dict[Any, Tuple[List[Any], List[Any]]] = {}  # per panel: the top and bottom bands' names
    for ax, (domain, columns) in zip(np.atleast_1d(axes), rows_by_domain.items()):
        if columns is None:
            _pending(ax, domain)
            continue
        drawn.append(ax)
        _style(ax)
        ax.spines["bottom"].set_visible(False)
        ax.set_xticks([])
        # Raw ELPD: every difference shifted by the best starting model's own.
        off = best_seed_elpd(domain, domain_data[domain])
        ax.axhline(off, color=INK_SECONDARY, linewidth=0.7, linestyle=(0, (3, 2)), zorder=0)
        xs = _column_positions(columns)
        pending_strips = []
        seed_names = _seed_names(columns)
        winners_here: List[Tuple[str, List[Tuple[float, float]]]] = []
        seeds_here: Dict[str, List[Tuple[float, float]]] = {}  # name -> its points, left to right
        for x, col in zip(xs, columns):
            if col["kind"] == "nonwinning":
                if col["values"] is None:
                    pending_strips.append(x)
                    continue
                # Left to right in the order each run admitted them.
                across = x + (np.asarray(col["progress"]) - 0.5) * (NONWINNING_SPAN - 0.5)
                _strip(ax, across, np.asarray(col["values"], dtype=float) + off, col["runs"])
                continue
            color = RUN_COLORS[col["run"]] if col["kind"] == "winner" else COLOR_SEED
            y = off + col["diff"]
            if col["se"] is None:
                hollow.append(f"{domain}:{col['model']}")
            _point(ax, (x, y), 2 * (col["se"] or 0.0), color, col["se"] is not None, vertical=True)
            if col["kind"] == "winner":  # named with its difference from the best starting model
                winners_here.append((f"{col['label']}  {col['diff']:+,.0f}", [(x, y)]))
            elif col["model"] in seed_names:
                seeds_here.setdefault(seed_names[col["model"]], []).append((x, y))
        bands[ax] = (winners_here, list(seeds_here.items()))
        if pending_strips:
            ax.text(float(np.mean(pending_strips)), off, "pending", ha="center", va="bottom",
                    fontsize=5, color=INK_SECONDARY, style="italic")
        # Rules between the starting models, the strips and the best discovered models.
        groups = ["winner" if c["kind"] == "winner" else "seed" if c["kind"] == "seed" else "strip"
                  for c in columns]
        lefts = [x - _column_width(c) / 2 for x, c in zip(xs, columns)]
        for i in range(1, len(columns)):
            if groups[i] != groups[i - 1]:
                ax.axvline(lefts[i], color=GRID, linewidth=0.6, zorder=0)
        ax.set_xlim(-0.2, xs[-1] + 0.7)
        ax.margins(y=0.08)
        ax.autoscale_view(scalex=False)
        ax.yaxis.set_major_locator(MaxNLocator(4, steps=[1, 2, 2.5, 5, 10]))
        _elpd_ticks(ax.yaxis)
        ax.tick_params(axis="y", labelsize=5.5, pad=1.5)
        ax.set_title(panel_title(domain, mock_domains=mock_domains), fontsize=7.5, color=INK,
                     loc="left", pad=3)
    drawn[0].set_ylabel("ELPD (nats)", fontsize=6.5, color=INK, labelpad=2)
    legend = _fig2_legend(fig, hollow, _n_runs(rows_by_domain))
    fig.tight_layout(w_pad=0.8, rect=(0, 0, 0.97, 1 - legend))
    fig.canvas.draw()  # the final layout, so names are placed in its coordinates
    for ax, (winners_here, seeds_here) in bands.items():
        _name_band(ax, fig, winners_here, "top")
        _name_band(ax, fig, seeds_here, "bottom")
    _save(fig, out_dir, "fig2_elpd")


HORIZONTAL_NONWINNING_SPAN = 2.4  # the non-winning strip's height in rows: its dots spread by progress
HORIZONTAL_PAPER_ROW = 1.5  # a starting model's row with its paper on a second line
FIG2_HORIZONTAL_HEIGHT_IN = 3.2


def _row_height(row: Dict[str, Any]) -> float:
    if row["kind"] == "nonwinning":
        return HORIZONTAL_NONWINNING_SPAN
    return HORIZONTAL_PAPER_ROW if row.get("paper") else 1.0


def _row_label(row: Dict[str, Any]) -> str:
    if row["kind"] == "nonwinning":
        return _strip_label(row)
    return f"{row['label']}\n({row['paper']})" if row.get("paper") else row["label"]


def draw_elpd_horizontal(domain_data: Dict[str, Optional[Dict[str, Any]]], out_dir: Path,
                         mock_domains: List[str], names: Dict[str, Dict[str, str]],
                         papers: Dict[str, Dict[str, Optional[str]]]) -> None:
    """``draw_elpd``'s comparison as a forest plot: per domain, side by side,
    one row per model, top to bottom the best discovered models, the other
    live models, the models pruned during the run (a band in which each
    rises with its progress through its run) and the starting models (with
    the paper each implements), ELPD across (right is a better fit; the best
    starting model's dashed). Every row is named on the axis."""
    rows_by_domain = {d: None if data is None else forest_rows(d, data, names[d], papers[d])
                      for d, data in domain_data.items()}
    fig, axes = plt.subplots(1, len(rows_by_domain), figsize=(FULL_WIDTH_IN, FIG2_HORIZONTAL_HEIGHT_IN))
    hollow: List[str] = []
    drawn = []
    for ax, (domain, rows) in zip(np.atleast_1d(axes), rows_by_domain.items()):
        if rows is None:
            _pending(ax, domain)
            continue
        drawn.append(ax)
        _style(ax)
        off = best_seed_elpd(domain, domain_data[domain])
        ax.axvline(off, color=INK_SECONDARY, linewidth=0.7, linestyle=(0, (3, 2)), zorder=0)
        heights = [_row_height(r) for r in rows]
        ys = list(np.cumsum(heights) - np.asarray(heights) / 2)
        for y, row in zip(ys, rows):
            if row["kind"] == "nonwinning":
                if row["values"] is None:
                    ax.text(off, y, " pending", ha="left", va="center", fontsize=5,
                            color=INK_SECONDARY, style="italic")
                    continue
                # Bottom to top in the order each run admitted them.
                across = y + (0.5 - np.asarray(row["progress"])) * (HORIZONTAL_NONWINNING_SPAN - 0.5)
                _strip(ax, np.asarray(row["values"], dtype=float) + off, across, row["runs"])
                continue
            color = RUN_COLORS[row["run"]] if row["kind"] == "winner" else COLOR_SEED
            x, half = off + row["diff"], 2 * (row["se"] or 0.0)
            if row["se"] is None:
                hollow.append(f"{domain}:{row['model']}")
            _point(ax, (x, y), half, color, row["se"] is not None, vertical=False)
            if row["kind"] == "winner":
                ax.annotate(f"{row['diff']:+,.0f}", (x + half, y), xytext=(3, 0),
                            textcoords="offset points", ha="left", va="center", fontsize=5,
                            color=INK, annotation_clip=False)
        # Rules between the winners, the strips and the starting models.
        groups = ["winner" if r["kind"] == "winner" else "seed" if r["kind"] == "seed" else "strip"
                  for r in rows]
        for i in range(1, len(rows)):
            if groups[i] != groups[i - 1]:
                ax.axhline(ys[i] - heights[i] / 2, color=GRID, linewidth=0.6, zorder=0)
        ax.set_yticks(ys, [_row_label(r) for r in rows], fontsize=5, linespacing=1.0)
        for tick, row in zip(ax.get_yticklabels(), rows):
            if row["kind"] == "nonwinning":
                tick.set_color(INK_SECONDARY)
                tick.set_style("italic")
        ax.tick_params(axis="y", length=0, pad=2)
        ax.set_ylim(ys[-1] + heights[-1] / 2 + 0.1, -0.1)
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo, hi + 0.15 * (hi - lo))  # room for the winners' differences
        ax.xaxis.set_major_locator(MaxNLocator(2, steps=[1, 2, 2.5, 5, 10]))
        _elpd_ticks(ax.xaxis)
        ax.tick_params(axis="x", labelsize=5.5, pad=1.5)
        ax.set_title(panel_title(domain, mock_domains=mock_domains), fontsize=7.5, color=INK,
                     loc="left", pad=3)
    drawn[len(drawn) // 2].set_xlabel("ELPD (nats)", fontsize=6.5, color=INK, labelpad=2)
    legend = _fig2_legend(fig, hollow, _n_runs(rows_by_domain))
    fig.tight_layout(w_pad=0.6, rect=(0, 0, 1, 1 - legend))
    _save(fig, out_dir, "fig2_elpd_horizontal")


# Figure 3's point opacity where a domain's points pile up: the number
# game's 840 cells all have 9 responses, so every point is full size and
# they stack on the ten possible observed proportions.
FIG3_ALPHA = {"number_game": 0.1}


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
            ax.scatter(predicted, observed, s=sizes, marker="o", color=color,
                       alpha=FIG3_ALPHA.get(domain, 0.25),
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
    names = yaml.safe_load(args.model_names.read_text())
    papers = yaml.safe_load(args.seed_papers.read_text())
    drawn = [d for d, data in domain_data.items() if data is not None]
    for draw in (draw_elpd, draw_elpd_horizontal):
        draw(domain_data, args.out_dir, args.mock_domains, {d: names[d] for d in drawn},
             {d: papers[d] for d in drawn})
    draw_conditions(domain_data, args.out_dir, args.mock_domains)
    draw_pseudocode(specs, args.out_dir, args.mock_domains)
    print(f"wrote figures 2-4 to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
