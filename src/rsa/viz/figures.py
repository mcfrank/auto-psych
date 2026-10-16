"""Figures for the RSA campaign: held-out tests, recovery, chain trajectories, the
prospective tests, the pre-registered test, model vs data, and the consolidation.

    uv run python -m src.rsa.viz.figures [--out docs/auto_rsa/figures]

Every figure is drawn from summaries already in the repository (no fitting):
- run 2's held-out tables and recovery verdicts (`data/rsa/sherlock_run2/`);
- each chain's loop histories and prospective tests (`data/rsa/live/live_rsa_c*/`);
- the pre-registered analysis (`data/rsa/prereg/run/prereg_analysis.json`);
- the consolidation (`data/rsa/consolidate/`).

The model-vs-data figure uses `data/rsa/viz/panels.json` (`src.rsa.viz.export_panels`, run on
Sherlock) when it exists. Without it, it is drawn from the chains' report bundles (observed
counts) and the frozen pool predictions, for the displays in the pool, and has no
pre-registered data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tyro  # noqa: E402
from pyprojroot import here  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
LIGHT = "#c9c8c1"

CHAINS = ("c0", "c1", "c2")
FINALS = {"c0": "boundary_avoidable_ambiguity_listener", "c1": "feature_surprisal_ensemble_listener",
          "c2": "competition_endpoint_binding_lapse_l2"}
BEST_PROMOTED = "promoted:rsa_l2_singleton_distinct_feat_color_l0"
BEST_STARTING = "seed:rsa_l1_shared_prior"
DATASET_COLORS = {"literature": BLUE, "campaign": ORANGE, "prereg": AQUA}
ROW_LABELS = {"literature": "published", "campaign": "campaign", "prereg": "pre-registered"}
DATASET_LABELS = {"literature": "published studies", "campaign": "campaign (9 live experiments)",
                  "prereg": "pre-registered experiment"}


@dataclass
class Args:
    out: Path = here() / "docs/auto_rsa/figures"
    panels: Path = here() / "data/rsa/viz/panels.json"
    """Model-vs-data export from Sherlock; the interim figure is drawn without it."""


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelcolor": INK2, "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 8.5,
        "text.color": INK, "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
    })


def load(path: str):
    return json.loads((here() / path).read_text())


def short(name: str) -> str:
    return name.split(":", 1)[-1]


def save(fig, out: Path, name: str) -> Path:
    path = out / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
    return path


def ci_row(ax, y, diff, se, color, size=7, lw=2.0, label=None):
    ax.plot([diff - 1.96 * se, diff + 1.96 * se], [y, y], color=color, lw=lw, solid_capstyle="round")
    ax.plot([diff], [y], "o", color=color, ms=size, mec=SURFACE, mew=1.5, label=label, zorder=3)


# ---------------------------------------------------------------- figure 1: run 2

def fig_existing_data(out: Path) -> Path:
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw=dict(width_ratios=[1.15, 1]))
    for i, rep in enumerate((1, 2, 3)):
        h = load(f"data/rsa/sherlock_run2/real_rep{rep}/heldout/heldout.json")
        best = load(f"data/rsa/sherlock_run2/real_rep{rep}/export.json")["best_model"]
        rows = {r["model"]: r for r in h["table"]}
        loop = [r for m, r in rows.items() if not m.startswith(("seed:", "pruned:"))]
        pruned = [r for m, r in rows.items() if m.startswith("pruned:") and short(m) not in
                  {short(s) for s in rows if s.startswith("seed:")}]
        seeds = [r for m, r in rows.items() if m.startswith("seed:")]
        x0 = i
        jit = np.random.default_rng(rep)
        a.scatter(x0 - 0.18 + jit.uniform(-0.07, 0.07, len(pruned)), [r["diff_vs_best_seed"] for r in pruned],
                  s=12, color=LIGHT, lw=0, label="pruned during the run" if i == 0 else None)
        a.scatter(x0 + 0.0 + jit.uniform(-0.07, 0.07, len(loop)), [r["diff_vs_best_seed"] for r in loop],
                  s=16, color=BLUE, alpha=0.45, lw=0, label="live at the end" if i == 0 else None)
        a.scatter([x0 - 0.36] * len(seeds), [r["diff_vs_best_seed"] for r in seeds], marker="_", s=120,
                  color=INK2, lw=1.5, label="starting models" if i == 0 else None)
        r = rows[best]
        a.errorbar([x0 + 0.24], [r["diff_vs_best_seed"]], yerr=[1.96 * r["se_diff_clustered"]], fmt="o", ms=8,
                   color=ORANGE, mec=SURFACE, mew=1.5, elinewidth=2, capsize=0,
                   label="exported model (95% CI)" if i == 0 else None, zorder=4)
        a.annotate(f"+{r['diff_vs_best_seed']:.0f}", (x0 + 0.24, r["diff_vs_best_seed"] + 1.96 * r["se_diff_clustered"]),
                   xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    a.axhline(0, color=INK2, lw=1)
    a.set_xticks(range(3), ["run 1", "run 2", "run 3"])
    a.set_ylabel("held-out log predictive density\nvs best starting model (nats)")
    a.set_ylim(-330, 170)
    a.set_title("A. Held-out published data: loop models vs the best starting model", fontsize=10)
    a.legend(loc="lower left", ncol=2)
    a.grid(axis="x", visible=False)

    cells = [("literal", 1), ("literal", 2), ("salience", 1), ("salience", 2)]
    for i, (gt, rep) in enumerate(cells):
        d = load(f"data/rsa/sherlock_run2/recovery_{gt}_rep{rep}/recovery/recovery.json")
        live = list(d["live_pool_rmse_to_gt"].values())
        seeds = list(d["seed_pool_rmse_to_gt"].values())
        b.scatter([i - 0.2] * len(seeds), seeds, marker="_", s=120, color=INK2, lw=1.5,
                  label="starting models" if i == 0 else None)
        b.scatter(i + np.random.default_rng(i).uniform(-0.06, 0.06, len(live)), live, s=16, color=BLUE, alpha=0.45,
                  lw=0, label="live at the end" if i == 0 else None)
        b.scatter([i + 0.2], [d["best_pool_rmse_to_gt"]], s=60, color=ORANGE, edgecolor=SURFACE, lw=1.5, zorder=4,
                  label="exported model" if i == 0 else None)
    b.axhline(0.01, color=INK2, lw=1, ls=(0, (4, 3)))
    b.text(3.45, 0.0108, "recovery threshold", ha="right", va="bottom", fontsize=8, color=INK2)
    b.set_yscale("log")
    b.set_xticks(range(4), ["literal\nrun 1", "literal\nrun 2", "salience\nrun 1", "salience\nrun 2"])
    b.set_ylabel("RMSE to the ground truth's predictions\n(1,597-display pool, log scale)")
    b.set_title("B. Recovery from simulated data: 4 of 4 recovered", fontsize=10)
    b.set_ylim(8e-4, 0.4)
    b.legend(loc="upper center", ncol=3)
    b.grid(axis="x", visible=False)
    fig.suptitle("Existing-data phase (run 2): the loop beats its starting models and recovers known truths",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return save(fig, out, "fig1_existing_data")


# ---------------------------------------------------------------- figure 2: chain trajectories

def chain_trajectory(chain: str, experiment: int) -> List[tuple]:
    steps = load(f"data/rsa/live/live_rsa_{chain}/outputs/experiment{experiment}/model_loop/history.json")
    out = []
    for s in steps:
        st = s["standing"][s["best_model"]]
        out.append((s["round"], st["sel_elpd"], s["best_model"]))
    return out


def fig_trajectories(out: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    colors = {1: BLUE, 2: ORANGE, 3: AQUA}
    for ax, chain in zip(axes, CHAINS):
        for e in (1, 2, 3):
            traj = chain_trajectory(chain, e)
            base = traj[0][1]
            xs = [r + 1 for r, _, _ in traj]
            ys = [v - base for _, v, _ in traj]
            ax.plot(xs, ys, color=colors[e], marker="o", ms=4.5, mec=SURFACE, mew=1,
                    label=f"experiment {e}")
            changes = [i for i in range(1, len(traj)) if traj[i][2] != traj[i - 1][2]]
            ax.scatter([xs[i] for i in changes], [ys[i] for i in changes], s=70, facecolor="none",
                       edgecolor=colors[e], lw=1.2, zorder=4)
            ax.annotate(f"+{ys[-1]:.0f}", (xs[-1], ys[-1]), xytext=(5, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=INK)
        ax.set_xticks(range(7), ["start", "1", "2", "3", "4", "5", "6"])
        ax.set_xlim(-0.3, 6.9)
        ax.set_xlabel("loop round")
        ax.set_title(f"chain {chain[1]}\n→ {FINALS[chain]}", fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("selection ELPD of the incumbent\nvs the experiment's starting set (nats)")
    axes[0].legend(loc="upper left")
    axes[2].scatter([], [], s=70, facecolor="none", edgecolor=INK2, label="new incumbent")
    axes[2].legend(handles=[axes[2].collections[-1]], loc="upper left")
    fig.suptitle("Within each live experiment, agents keep finding better models (grouped-CV ELPD on the "
                 "data so far)", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return save(fig, out, "fig2_chain_trajectories")


# ---------------------------------------------------------------- figure 3: prospective tests

def fig_prospective(out: Path) -> Path:
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw=dict(width_ratios=[1, 1.05]))
    rows = []
    for chain in CHAINS:
        for e in (2, 3):
            p = load(f"data/rsa/live/live_rsa_{chain}/private/experiment{e}/prospective.json")
            rows.append((f"chain {chain[1]}, exp {e}", p["committed_vs"]))
    for i, (label, cv) in enumerate(rows):
        y = len(rows) - 1 - i
        ci_row(a, y + 0.14, cv["best_seed"]["diff"], cv["best_seed"]["se"], INK2, size=6, lw=1.6,
               label="vs best starting model" if i == 0 else None)
        ci_row(a, y - 0.14, cv["best_promoted"]["diff"], cv["best_promoted"]["se"], BLUE,
               label="vs best promoted seed" if i == 0 else None)
    a.axvline(0, color=INK2, lw=1)
    a.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
    a.set_xlabel("log predictive density difference on the next experiment (nats, 95% CI)")
    a.set_title("A. Committed model, scored on the next experiment", fontsize=10)
    a.set_xlim(-55, 345)
    a.legend(loc="lower right")
    a.grid(axis="y", visible=False)

    d = load("data/rsa/prereg/run/prereg_analysis.json")
    order = ["c0", "c2", "c1"]
    for i, chain in enumerate(order):
        f = FINALS[chain]
        y = len(order) - 1 - i
        prim = d["primary"][f]
        ci_row(b, y + 0.2, prim["diff"], prim["se"], BLUE if prim["holm_reject"] else MUTED, size=8, lw=2.4,
               label="all 312 displays" if i == 0 else None)
        for k, (half, mk) in enumerate((("eig", "s"), ("random", "D"))):
            h = d["by_half"][half]["primary"][f]
            yy = y - 0.08 - 0.16 * k
            b.plot([h["diff"] - 1.96 * h["se"], h["diff"] + 1.96 * h["se"]], [yy, yy], color=LIGHT, lw=1.6)
            b.plot([h["diff"]], [yy], mk, color=INK2, ms=4.5, mec=SURFACE, mew=1,
                   label=("EIG half" if half == "eig" else "random half") if i == 0 else None)
        verdict = "rejects H0" if prim["holm_reject"] else "does not reject"
        b.annotate(f"{prim['diff']:+.0f} ± {prim['se']:.0f}, z = {prim['z']:.2f}\nHolm: {verdict}",
                   (prim["diff"] + 1.96 * prim["se"], y + 0.2), xytext=(6, 0), textcoords="offset points",
                   va="center", fontsize=8, color=INK)
    b.axvline(0, color=INK2, lw=1)
    b.set_yticks(range(3), [f"chain {c[1]}'s final" for c in reversed(order)])
    b.set_xlim(-60, 380)
    b.set_xlabel(f"lpd difference vs the best promoted seed (nats, 95% CI)\n"
                 f"{d['n_participants']:,} new people, {d['n_trials']:,} trials")
    b.set_title("B. Pre-registered test (OSF s48jv)", fontsize=10)
    b.legend(loc="lower right")
    b.grid(axis="y", visible=False)
    fig.suptitle("Prospective tests: the agents' models predict new people better than the models they "
                 "started from", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return save(fig, out, "fig3_prospective_tests")


# ---------------------------------------------------------------- figure 4: every model on the prereg data

def fig_prereg_ranking(out: Path) -> Path:
    d = load("data/rsa/prereg/run/prereg_analysis.json")
    lpd = dict(d["lpd"])
    lpd.pop("promoted:rsa_l2", None)
    lpd.pop("promoted:rsa_l1", None)  # duplicates of the seeds
    ref = lpd[BEST_PROMOTED]
    items = sorted(lpd.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    finals = {v: k for k, v in FINALS.items()}
    for y, (m, v) in enumerate(items):
        if m in finals:
            color, kind = ORANGE, "final"
        elif m.startswith("promoted:"):
            color, kind = BLUE, "promoted"
        else:
            color, kind = INK2, "seed"
        ax.plot([0, v - ref], [y, y], color=color, lw=2, alpha=0.35)
        ax.plot([v - ref], [y], "o", color=color, ms=7, mec=SURFACE, mew=1.5)
        name = short(m) + (f"  (chain {finals[m][1]} final)" if m in finals else "")
        ax.annotate(name, (v - ref, y), xytext=(7 if v >= ref else -7, 0), textcoords="offset points",
                    ha="left" if v >= ref else "right", va="center", fontsize=8,
                    color=INK if m in finals or m == BEST_PROMOTED else INK2)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks([])
    ax.set_xlim(-1550, 650)
    ax.set_xlabel("log predictive density vs the best promoted seed (nats; 14,404 trials)")
    for color, label in ((ORANGE, "chain finals"), (BLUE, "promoted seeds (run 2's best, refitted)"),
                         (INK2, "starting models (textbook RSA and variants)")):
        ax.plot([], [], "o", color=color, ms=7, label=label)
    ax.legend(loc="upper left")
    ax.grid(axis="y", visible=False)
    ax.set_title("Every model on the pre-registered experiment, predictions frozen before data", fontsize=11)
    fig.tight_layout()
    return save(fig, out, "fig4_prereg_all_models")


# ---------------------------------------------------------------- figure 5: model vs data

def interim_panels() -> dict:
    """Observed counts from the chains' report bundles, predictions from the frozen pool files,
    for the bundle displays that are in the pool (no pre-registered data)."""
    finals = load("data/rsa/live/finals_pool_preds_final.json")
    bar = load("data/rsa/live/bar_pool_preds.json")
    pool = finals["pool"]
    offsets = np.concatenate([[0], np.cumsum(pool["n_entries"])])
    index = {}
    for i, c in enumerate(pool["contexts"]):
        if c["valence"] in (0, None) and c["familiarization"] is None and c["grayscale"] is None:
            index[(json.dumps(c["objects"]), c["utterance"])] = i
    sources = {"c0": np.asarray(finals["models"][FINALS["c0"]]["probs"]),
               "promoted": np.asarray(bar["models"][BEST_PROMOTED]["probs"]),
               "starting": np.asarray(bar["models"][BEST_STARTING]["probs"])}
    literature = ("size", "E1_dv", "E2_manip_check", "E3_ling_frame", "E4_prior_frame", "E8_levels", "E9_twins",
                  "E10_oddman", "levels_prior_action")
    panels, seen_lit = [], set()
    for chain in CHAINS:
        bundle = load(f"data/rsa/live/live_rsa_{chain}/outputs/experiment3/model_loop/report.bundle.json")
        for p in bundle["panels"]:
            campaign = p["experiment"].startswith("outer_e")
            if not campaign and (p["experiment"] not in literature or (p["experiment"], p["condition"]) in seen_lit):
                continue
            utt = p["utterance"] if p["query"] == "utterance" else None
            i = index.get((json.dumps(p["objects"]), utt))
            if i is None:
                continue
            if not campaign:
                seen_lit.add((p["experiment"], p["condition"]))
            panels.append(dict(
                dataset="campaign" if campaign else "literature", experiment=p["experiment"], n=p["n"],
                classes=[dict(cls=c["object"], copies=c["copies"], count=c["count"],
                              preds={k: float(v[offsets[i] + c["object"]]) for k, v in sources.items()})
                         for c in p["classes"]]))
    return dict(panels=panels, interim=True)


def fig_model_vs_data(out: Path, panels_path: Path) -> Path:
    if panels_path.exists():
        data = json.loads(panels_path.read_text())
        for p in data["panels"]:
            p["dataset"] = "campaign" if p["dataset"].startswith("campaign") else p["dataset"]
        interim = False
    else:
        data, interim = interim_panels(), True
    models = [("c0", "chain 0's final\n(boundary_avoidable_ambiguity_listener)"),
              ("promoted", "best promoted seed\n(rsa_l2_singleton_distinct_feat_color_l0)"),
              ("starting", "best starting model\n(rsa_l1_shared_prior)")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True)
    datasets = [d for d in ("literature", "campaign", "prereg") if any(p["dataset"] == d for p in data["panels"])]
    for ax, (label, title) in zip(axes, models):
        ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
        stats = []
        for d in datasets:
            pts = [(c["preds"][label], c["count"] / p["n"], p["n"]) for p in data["panels"] if p["dataset"] == d
                   for c in p["classes"] if p["n"] > 0]
            x, y, n = (np.asarray(v) for v in zip(*pts))
            ax.scatter(x, y, s=4 + 1.6 * np.sqrt(n), color=DATASET_COLORS[d], alpha=0.55, lw=0.4,
                       edgecolor=SURFACE, label=DATASET_LABELS[d], zorder=2 if d != "prereg" else 3)
            r = np.corrcoef(x, y)[0, 1]
            rmse = np.sqrt(np.average((x - y) ** 2, weights=n))
            stats.append(f"{d}: r = {r:.3f}, RMSE = {rmse:.3f}")
        ax.text(0.02, 0.98, "\n".join(stats), transform=ax.transAxes, va="top", fontsize=7.8, color=INK2)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.set_xlabel("model's predicted choice probability")
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("observed choice proportion")
    axes[0].legend(loc="lower right", markerscale=1.3)
    sub = ("one point per display × object class, size by number of responses; models fitted to the published "
           "and campaign data, the pre-registered data never fitted")
    if interim:
        sub = ("INTERIM: pool displays only, no pre-registered data yet (needs data/rsa/viz/panels.json); "
               "one point per display × object class, size by responses")
    fig.suptitle("Model vs data across conditions", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.01, 0.905, sub, fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return save(fig, out, "fig5_model_vs_data")


def fig_model_vs_data_by_dataset(out: Path, panels_path: Path) -> Optional[Path]:
    """Figure 5 split into one row per dataset (needs the Sherlock export)."""
    if not panels_path.exists():
        return None
    data = json.loads(panels_path.read_text())
    models = [("c0", "chain 0's final"), ("promoted", "best promoted seed"), ("starting", "best starting model")]
    datasets = ["literature", "campaign", "prereg"]
    fig, axes = plt.subplots(3, 3, figsize=(10, 10.4), sharex=True, sharey=True)
    for row, d in enumerate(datasets):
        panels = [p for p in data["panels"] if (p["dataset"].startswith("campaign") and d == "campaign")
                  or p["dataset"] == d]
        for col, (label, title) in enumerate(models):
            ax = axes[row, col]
            ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
            pts = [(c["preds"][label], c["count"] / p["n"], p["n"]) for p in panels for c in p["classes"] if p["n"] > 0]
            x, y, n = (np.asarray(v) for v in zip(*pts))
            ax.scatter(x, y, s=3 + 1.4 * np.sqrt(n), color=DATASET_COLORS[d], alpha=0.55, lw=0.4, edgecolor=SURFACE,
                       zorder=2)
            r = np.corrcoef(x, y)[0, 1]
            rmse = np.sqrt(np.average((x - y) ** 2, weights=n))
            head = f"{ROW_LABELS[d]}, {len(panels)} displays\n" if col == 0 else ""
            ax.text(0.03, 0.97, f"{head}r = {r:.3f}\nRMSE = {rmse:.3f}", transform=ax.transAxes, va="top",
                    fontsize=8.5, color=INK)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_aspect("equal")
            if row == 0:
                ax.set_title(title, fontsize=10)
            if row == 2:
                ax.set_xlabel("predicted probability")
            if col == 0:
                ax.set_ylabel("observed proportion")
    fig.suptitle("Model vs data, one row per dataset", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.01, 0.945, "models fitted to the published and campaign data; the pre-registered row is "
             "out of sample", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save(fig, out, "fig5b_model_vs_data_by_dataset")


# ---------------------------------------------------------------- figure 6: consolidation

def fig_consolidation(out: Path) -> Path:
    e = load("data/rsa/consolidate/eliminate/elimination.json")
    s = load("data/rsa/consolidate/score.json")
    kept = set(e["rounds"][-1]["mechanisms"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
    series: Dict[str, List[tuple]] = {}
    for r, rd in enumerate(e["rounds"], start=1):
        for c in rd["candidates"]:
            series.setdefault(c["mechanism"], []).append((r, c["cost_in_se"], c["converged"]))
    dropped_at, ends = {}, {}
    for r, rd in enumerate(e["rounds"][:-1], start=1):
        gone = set(rd["mechanisms"]) - set(e["rounds"][r]["mechanisms"])
        for m in gone:
            dropped_at[m] = r
    a.axhspan(-2, 2, color=GRID, alpha=0.5, lw=0)
    a.axhline(2, color=INK2, lw=1, ls=(0, (4, 3)))
    a.text(7.45, 2.1, "2 SE: drop if below", ha="right", va="bottom", fontsize=8, color=INK2)
    for m, pts in series.items():
        color = BLUE if m in kept else ORANGE
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        a.plot(xs, ys, color=color, lw=1.4, alpha=0.75)
        a.scatter(xs, ys, s=14, color=[color if p[2] else SURFACE for p in pts], edgecolor=color, lw=1, zorder=3)
        if m in dropped_at:
            r = dropped_at[m]
            y = dict((p[0], p[1]) for p in pts)[r]
            a.scatter([r], [y], marker="X", s=60, color=ORANGE, edgecolor=SURFACE, lw=0.8, zorder=4)
            a.annotate(m, (r, y), xytext=(0, -9), textcoords="offset points", ha="center", va="top", fontsize=7.5,
                       color=INK2)
        else:
            ends[m] = ys[-1]
    placed = []
    for m, y in sorted(ends.items(), key=lambda kv: kv[1]):
        y_text = max(y, placed[-1] + 0.42) if placed else y
        placed.append(y_text)
        a.annotate(m, (len(e["rounds"]), y), xytext=(len(e["rounds"]) + 0.15, y_text), textcoords="data",
                   va="center", fontsize=7.8, color=INK)
    a.plot([], [], color=BLUE, label="kept in the lean model")
    a.plot([], [], color=ORANGE, marker="X", ms=7, mec=SURFACE, label="dropped (× = round dropped)")
    a.scatter([], [], s=14, color=SURFACE, edgecolor=MUTED, label="fit not converged")
    a.legend(loc="upper left", ncol=1)
    a.set_xlim(0.5, 8.4)
    a.set_xticks(range(1, 8))
    a.set_xlabel("elimination round")
    a.set_ylabel("cost of removing the mechanism\n(grouped-CV ELPD, in clustered SEs)")
    a.set_title("A. Backward elimination of the 13-mechanism superset (pooled data)", fontsize=10)
    a.grid(axis="x", visible=False)

    order = ["lean", "full", "c2", "c1", "promoted"]
    names = {"lean": "lean consolidated model", "full": "full superset (unconverged)",
             "c2": "chain 2's final", "c1": "chain 1's final", "promoted": "best promoted seed"}
    for i, k in enumerate(order):
        m = s["models"][k]
        y = len(order) - 1 - i
        vs = m["vs_reference"]
        ci_row(b, y, vs["diff"], vs["se"], ORANGE if k == "lean" else (MUTED if k == "full" else BLUE),
               size=8 if k == "lean" else 6.5)
        b.annotate(f"{m['n_params']} params", (vs["diff"] + 1.96 * vs["se"], y), xytext=(6, 0),
                   textcoords="offset points", va="center", fontsize=8, color=INK2)
    b.axvline(0, color=INK2, lw=1)
    b.text(-4, len(order) - 0.45, "chain 0's final (17 params)", ha="right", fontsize=8, color=INK2)
    b.set_yticks(range(len(order)), [names[k] for k in reversed(order)])
    b.set_xlim(-290, 60)
    b.set_ylim(-0.6, len(order) - 0.3)
    b.set_xlabel("lpd vs chain 0's final on the pre-registered data (nats, 95% CI)")
    b.set_title("B. Scored prospectively: the lean model ranks second", fontsize=10)
    b.grid(axis="y", visible=False)
    fig.suptitle("Consolidating the three finals: 7 mechanisms, 9 parameters", x=0.01, ha="left", fontsize=12,
                 fontweight="bold")
    fig.tight_layout()
    return save(fig, out, "fig6_consolidation")


def main(args: Args) -> List[Path]:
    style()
    args.out.mkdir(parents=True, exist_ok=True)
    return [fig_existing_data(args.out), fig_trajectories(args.out), fig_prospective(args.out),
            fig_prereg_ranking(args.out), fig_model_vs_data(args.out, args.panels),
            *filter(None, [fig_model_vs_data_by_dataset(args.out, args.panels)]), fig_consolidation(args.out)]


if __name__ == "__main__":
    main(tyro.cli(Args))
