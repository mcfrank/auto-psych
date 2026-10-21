"""The paper's Figures 2-4, one column (or row) per domain.

- Figure 2 (``fig2_elpd``): per domain, a forest plot of ELPD differences
  from the best starting model, the runs collapsed (subjective randomness:
  all runs pooled; reference games: every chain on the pre-registered test,
  against the best promoted seed; the number game's one run): the best
  discovered models and the starting models as named points with ±2 SE of
  the paired difference (names from ``model_names.yaml``), the other models
  live at the end and the models pruned during the runs as strips. How each
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
from matplotlib.ticker import MaxNLocator  # noqa: E402
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
    model_names: Path = here() / "scripts" / "paper" / "model_names.yaml"
    """Human-readable names of the models Figure 2 labels, per domain."""
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


def _model_name(names: Dict[str, str], domain: str, model: str) -> str:
    if model not in names:
        raise KeyError(f"{domain}: {model!r} has no name in model_names.yaml")
    return names[model]


def _in_run_order(domain: str, by_run: Dict[str, Dict[str, float]],
                  order: Dict[str, int]) -> Tuple[List[float], List[float]]:
    """The pruned models' values and their progress through their own run
    (rank in admission order, scaled to 0..1), run by run."""
    values: List[float] = []
    progress: List[float] = []
    for run, models in by_run.items():
        missing = sorted(set(models) - set(order))
        if missing:
            raise RuntimeError(f"{domain} {run}: no admission order for {missing}")
        ranked = sorted(models, key=order.get)
        for i, label in enumerate(ranked):
            values.append(models[label])
            progress.append(i / (len(ranked) - 1) if len(ranked) > 1 else 0.5)
    return values, progress


def forest_rows(domain: str, data: Dict[str, Any], names: Dict[str, str]) -> List[Dict[str, Any]]:
    """Figure 2's rows for one domain, top to bottom: the best discovered
    models (named, with their run when there are several), the other models
    live at the end and the models pruned during the runs (unnamed strips),
    then the starting models (named), each against the best starting model.

    A labelled row has ``diff`` and ``se`` (``None``: its paired SE is not
    computed yet; drawn hollow, without a CI). A strip has ``values``, or
    ``None`` when its models are not scored yet (drawn as "pending"); the
    pruned strip also has each model's ``progress`` through its own run, 0
    for the first it admitted to 1 for the last (``pruned_order``, ranked
    within each run, so collapsed runs overlay)."""
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
        others = list(pooled["others_at_end"].values()) if scored else None
        pruned = progress = None
        if scored:
            by_run = {run: {label: v for label, v in pooled["pruned_during_run"].items()
                            if label.startswith(f"{run}__")} for run in pooled["runs"]}
            pruned, progress = _in_run_order(domain, by_run, pooled["pruned_order"])
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
        others = [v for r in runs for v in r["others_at_end"].values()] if scored else None
        pruned = progress = None
        if scored:
            order = {f"{r['run']}__{n}": i for r in runs for n, i in r["pruned_order"].items()}
            by_run = {r["run"]: {f"{r['run']}__{n}": v for n, v in r["pruned_during_run"].items()}
                      for r in runs}
            pruned, progress = _in_run_order(domain, by_run, order)

    several = len(winners) > 1
    rows = [{"kind": "winner", "model": m,
             "label": _model_name(names, domain, m)
             + (f" ({run_label(domain, run).split()[-1]})" if several else ""),
             **value(m)} for m, run in winners]
    rows.sort(key=lambda r: -r["diff"])
    rows.append({"kind": "others", "label": "Other live models", "values": others})
    rows.append({"kind": "pruned", "label": "Pruned during the run", "values": pruned,
                 "progress": progress})
    seed_rows = [{"kind": "seed", "model": m, "label": _model_name(names, domain, m), **value(m)}
                 for m in seeds]
    return rows + sorted(seed_rows, key=lambda r: -r["diff"])


PRUNED_SPAN = 3.0  # the pruned strip's width in columns: its dots spread by progress
FIG2_HEIGHT_IN = 2.3


SEED_WIDTH = 0.4  # a starting model's column width (the others' is 1)


def _column_width(column: Dict[str, Any]) -> float:
    return {"pruned": PRUNED_SPAN, "seed": SEED_WIDTH}.get(column["kind"], 1.0)


def _column_positions(columns: List[Dict[str, Any]]) -> List[float]:
    """Each column's centre, left to right (``_column_width`` each)."""
    centres, edge = [], 0.0
    for column in columns:
        width = _column_width(column)
        centres.append(edge + width / 2)
        edge += width
    return centres


def _strip_label(row: Dict[str, Any]) -> str:
    name = "Pruned during the run" if row["kind"] == "pruned" else "Other live models"
    return name if row["values"] is None else f"{name} ({len(row['values'])})"


NAME_GAP_PT = 1.5  # the least gap around a name, in points
MARK_RADIUS_PT = 2.5  # the space a point or dot takes, in points


def _place_names(ax, fig, names: List[Tuple[str, float, float, str]],
                 marks: List[Tuple[float, float, float]], texts: Sequence[Any] = (),
                 boxes: Sequence[Any] = ()) -> None:
    """Write each (text, x, y, side) name beside its point, horizontally: to
    the left of it (``side`` "left") or to the left just above it
    ("above-left", for the best starting model on the zero line: the starting
    models are all at or below it, so the space above them is free), tried
    upwards first, at the nearest height where it covers no mark
    (``marks``: each (x, low, high) point, CI or dot of the panel) and no
    other name or text (``texts``); on each side the names keep their
    points' top-to-bottom order. A name moved off its point gets a thin
    leader line back to it."""
    from matplotlib.transforms import Bbox

    renderer = fig.canvas.get_renderer()
    to_px = fig.dpi / 72.0
    r, gap = MARK_RADIUS_PT * to_px, NAME_GAP_PT * to_px
    taken = []
    for x, low, high in marks:
        (px, p_low), (_, p_high) = ax.transData.transform([(x, low), (x, high)])
        taken.append(Bbox([[px - r, p_low - r], [px + r, p_high + r]]))
    taken += [text.get_window_extent(renderer) for text in texts]
    taken += list(boxes)
    frame = ax.get_window_extent(renderer)
    lowest: Dict[str, float] = {}  # each side's lowest name so far (its bottom edge)

    def annotate(text, x, y, side, dy, leader):
        name = ax.annotate(
            text, (x, y), xytext=(-2 if side == "above-left" else 2, 4 + dy),
            textcoords="offset points", ha="right" if side == "above-left" else "left",
            va="bottom", fontsize=5, color=INK,
            annotation_clip=False, zorder=4,
            arrowprops=(dict(arrowstyle="-", color=INK_SECONDARY, linewidth=0.4, shrinkA=0,
                             shrinkB=2.5, relpos=(1.0, 0.0) if side == "above-left" else (0.0, 0.0))
                        if leader else None))
        name.set_in_layout(False)
        return name

    for text, x, y, side in sorted(names, key=lambda n: -n[2]):
        trial = annotate(text, x, y, side, 0, False)
        base = trial.get_window_extent(renderer)
        trial.remove()
        if side == "above-left" and base.x0 < frame.x0:  # it would cross the y-axis
            side = "above-right"
            trial = annotate(text, x, y, side, 0, False)
            base = trial.get_window_extent(renderer)
            trial.remove()
        first = 1
        lowest.setdefault(side, frame.y1)
        shift = 0.0
        for step in range(0, 120):
            found = False
            for sign in ((first, -first) if step else (1,)):
                box = base.translated(0, sign * step * to_px)
                padded = Bbox([[box.x0 - gap, box.y0 - gap], [box.x1 + gap, box.y1 + gap]])
                if box.y0 < frame.y0 or box.y1 > frame.y1:
                    continue
                if box.y1 > lowest[side] - gap:
                    continue  # keep the side's names in their points' order (and below the band)
                if any(padded.overlaps(other) for other in taken):
                    continue
                shift, found = sign * step, True
                break
            if found:
                break
        annotate(text, x, y, side, shift, True)  # several points can share its height
        taken.append(base.translated(0, shift * to_px))
        lowest[side] = taken[-1].y0


NAME_LINE_PT = 7.0  # one line of the winners' header band


def _name_winners(ax, fig, winners: List[Tuple[str, float, float]]) -> List[Any]:
    """Name the best discovered models in a band at the top of the panel,
    best on top, each name ending at its point's column (or further right,
    where it would cross the y-axis) with a leader line down to it; the
    panel's range grows to hold the band, whose ticks are hidden. Returns
    what the names take up, as display boxes (the texts and the lines)."""
    from matplotlib.text import Text
    from matplotlib.transforms import Bbox

    height_pt = ax.bbox.height * 72.0 / fig.dpi
    band_pt = len(winners) * NAME_LINE_PT + 4.0
    lo, hi = ax.get_ylim()
    ax.set_yticks([v for v in ax.get_yticks() if lo <= v <= hi])
    share = band_pt / height_pt
    top = (hi - lo * share) / (1 - share)
    ax.set_ylim(lo, top)
    per_pt = (top - lo) / height_pt
    renderer = fig.canvas.get_renderer()
    frame = ax.get_window_extent(renderer)
    to_data = ax.transData.inverted()
    taken = []
    for k, (text, x, y) in enumerate(sorted(winners, key=lambda w: -w[2])):
        y_text = top - (k + 0.5) * NAME_LINE_PT * per_pt
        probe = ax.text(x, y_text, text, ha="right", va="center", fontsize=5)
        box = probe.get_window_extent(renderer)
        probe.remove()
        x_text = x if box.x0 >= frame.x0 else to_data.transform((box.x1 + frame.x0 - box.x0, 0))[0]
        name = ax.annotate(text, (x, y), xytext=(x_text, y_text), textcoords="data", ha="right",
                           va="center", fontsize=5, color=INK, annotation_clip=False, zorder=4,
                           arrowprops=dict(arrowstyle="-", color=INK_SECONDARY, linewidth=0.4,
                                           shrinkA=0, shrinkB=2.5, relpos=(1.0, 0.5)))
        name.set_in_layout(False)
        taken.append(Text.get_window_extent(name, renderer))
        (px, p_point), (qx, p_text) = ax.transData.transform([(x, y), (x_text, y_text)])
        taken.append(Bbox([[min(px, qx) - 1, p_point], [max(px, qx) + 1, p_text]]))
    return taken


def draw_elpd(domain_data: Dict[str, Optional[Dict[str, Any]]], out_dir: Path,
              mock_domains: List[str], names: Dict[str, Dict[str, str]]) -> None:
    """Per domain, side by side, ELPD differences from the best starting model
    (up is a better fit), left to right: the starting models (worst first),
    the models pruned during the run (left to right in the order each run
    admitted them), the other live models, and the best discovered models;
    starting and best discovered models as points with ±2 SE. Only the best
    discovered models and the best starting model are named, beside their
    points (the others are for the caption). Thin rules separate the groups."""
    rows_by_domain = {d: None if data is None else forest_rows(d, data, names[d])[::-1]
                      for d, data in domain_data.items()}
    widths = [6.0 if cols is None else _column_positions(cols)[-1] + 0.5
              for cols in rows_by_domain.values()]
    fig, axes = plt.subplots(1, len(widths), figsize=(FULL_WIDTH_IN, FIG2_HEIGHT_IN),
                             gridspec_kw={"width_ratios": widths})
    jitter = np.random.default_rng(0)
    hollow = []
    drawn = []
    to_name: Dict[Any, List[Tuple[str, float, float, str]]] = {}
    marks_by_ax: Dict[Any, List[Tuple[float, float, float]]] = {}
    texts_by_ax: Dict[Any, List[Any]] = {}
    winners_by_ax: Dict[Any, List[Tuple[str, float, float]]] = {}
    for ax, (domain, columns) in zip(np.atleast_1d(axes), rows_by_domain.items()):
        if columns is None:
            _pending(ax, domain)
            continue
        drawn.append(ax)
        _style(ax)
        ax.spines["bottom"].set_visible(False)
        ax.set_xticks([])
        ax.axhline(0.0, color=INK_SECONDARY, linewidth=0.8, zorder=0)
        xs = _column_positions(columns)
        pending_strips = []
        best_seed = max((c for c in columns if c["kind"] == "seed"), key=lambda c: c["diff"])
        names_here = []
        winners_here: List[Tuple[str, float, float]] = []
        marks_here: List[Tuple[float, float, float]] = []
        for x, col in zip(xs, columns):
            if col["kind"] in ("others", "pruned"):
                if col["values"] is None:
                    pending_strips.append(x)
                    continue
                values = np.asarray(col["values"], dtype=float)
                if col["kind"] == "pruned":
                    # Left to right in the order each run admitted them.
                    across = x + (np.asarray(col["progress"]) - 0.5) * (PRUNED_SPAN - 0.5)
                    ax.scatter(across, values, s=5, color=COLOR_PRUNED, linewidths=0,
                               alpha=0.85, zorder=2)
                else:
                    across = x + jitter.uniform(-0.15, 0.15, values.size)
                    ax.scatter(across, values, s=8, color=COLOR_OTHERS, linewidths=0,
                               alpha=0.85, zorder=2)
                marks_here += [(float(a), float(v), float(v)) for a, v in zip(across, values)]
                continue
            color = COLOR_DISCOVERED if col["kind"] == "winner" else COLOR_SEED
            half = 2 * (col["se"] or 0.0)
            if col["se"] is None:
                hollow.append(f"{domain}:{col['model']}")
                ax.plot(x, col["diff"], "o", markersize=3.6, markerfacecolor="white",
                        markeredgecolor=color, markeredgewidth=0.9, zorder=3)
            else:
                ax.errorbar(x, col["diff"], yerr=half, fmt="o", markersize=3.6, color=color,
                            ecolor=color, elinewidth=1.0, capsize=0, markeredgecolor="white",
                            markeredgewidth=0.4, zorder=3)
            marks_here.append((x, col["diff"] - half, col["diff"] + half))
            if col["kind"] == "winner":
                winners_here.append((f"{col['label']}  {col['diff']:+,.0f}", x, col["diff"]))
            elif col is best_seed:
                names_here.append((col["label"], x, col["diff"], "above-left"))
        to_name[ax] = names_here
        winners_by_ax[ax] = winners_here
        marks_by_ax[ax] = marks_here
        texts_by_ax[ax] = []
        if pending_strips:
            texts_by_ax[ax].append(ax.text(float(np.mean(pending_strips)), 0.0, "pending",
                                           ha="center", va="bottom", fontsize=5,
                                           color=INK_SECONDARY, style="italic"))
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
        ax.tick_params(axis="y", labelsize=5.5, pad=1.5)
        ax.set_title(panel_title(domain, mock_domains=mock_domains), fontsize=7.5, color=INK,
                     loc="left", pad=3)
    drawn[0].set_ylabel("ΔELPD vs. best starting model (nats)", fontsize=6.5, color=INK,
                        labelpad=2)
    handles = [
        Line2D([], [], marker="o", color=COLOR_SEED, linestyle="none", markersize=3.6,
               label="Starting models"),
        Line2D([], [], marker="o", color=COLOR_PRUNED, linestyle="none", markersize=2.6,
               label="Pruned during the run (left to right as tried)"),
        Line2D([], [], marker="o", color=COLOR_OTHERS, linestyle="none", markersize=3,
               label="Other live models"),
        Line2D([], [], marker="o", color=COLOR_DISCOVERED, linestyle="none", markersize=3.6,
               label="Best discovered (±2 SE)"),
    ]
    fig.legend(handles=handles, loc="upper left", ncol=len(handles), frameon=False, fontsize=5.5,
               bbox_to_anchor=(0.0, 1.0), handlelength=1.0, handletextpad=0.4, columnspacing=1.2)
    if hollow:
        print(f"[fig2] drawn without a CI (paired SE not computed yet): {hollow}", flush=True)
        fig.text(0.995, 0.99, "Open markers: paired SE not yet computed.", ha="right", va="top",
                 fontsize=5, color=INK_SECONDARY)
    fig.tight_layout(w_pad=0.8, rect=(0, 0, 0.97, 0.93))
    fig.canvas.draw()  # the final layout, so names are placed in its coordinates
    for ax, names_here in to_name.items():
        band = _name_winners(ax, fig, winners_by_ax[ax])
        fig.canvas.draw()
        _place_names(ax, fig, names_here, marks_by_ax[ax], texts_by_ax[ax], band)
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
    names = yaml.safe_load(args.model_names.read_text())
    draw_elpd(domain_data, args.out_dir, args.mock_domains,
              {d: names[d] for d, data in domain_data.items() if data is not None})
    draw_conditions(domain_data, args.out_dir, args.mock_domains)
    draw_pseudocode(specs, args.out_dir, args.mock_domains)
    print(f"wrote figures 2-4 to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
