#!/usr/bin/env python3
"""Draw the auto-psych automated cognitive-modeling loop as a labeled diagram.

Renders the two nested loops the repo is built around:

- OUTER LOOP (experiment loop): the five sequential stages in
  ``src/pipelines/outer_loop`` (theory -> design -> implement -> collect ->
  model loop), repeated once per experiment.
- INNER LOOP (cognitive-model improvement): what stage 5 runs over pooled data
  in ``src/pipelines/inner_loop`` (seed -> fit/score -> propose -> re-score ->
  export best model).

Writes both an SVG (vector) and a PNG (raster) into ``diagrams/`` at the repo
root. This is a documentation helper, not part of the pipeline.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # file output only, no display needed
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pyprojroot import here

# ── Palette ────────────────────────────────────────────────────────────────
AGENT_FC, AGENT_EC = "#cfe3ff", "#2b6cb0"  # Claude Code (LLM) agent stages
PROG_FC, PROG_EC = "#d4efd4", "#2f855a"  # programmatic (no-agent) stages
OUTER_FC, OUTER_EC = "#f6f3ff", "#7c5cbf"  # outer-loop container
INNER_FC, INNER_EC = "#fff6ec", "#c0791f"  # inner-loop container
FLOW = "#d9534f"  # loop / hand-off arrows

TITLE_FS, BODY_FS = 10.5, 8.6


def draw_box(ax, cx, cy, w, h, title, body, fc, ec):
    """One stage box: bold title line over a wrapped body description."""
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle="round,pad=0.3,rounding_size=1.2",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.8,
        )
    )
    ax.text(cx, cy + h / 2 - 1.7, title, ha="center", va="center",
            fontsize=TITLE_FS, fontweight="bold", color=ec)
    ax.text(cx, cy - 0.8, body, ha="center", va="center",
            fontsize=BODY_FS, color="#1a1a1a", linespacing=1.25)


def straight_arrow(ax, x1, y1, x2, y2, color="#444", lw=1.8):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                shrinkA=0, shrinkB=0))


def elbow(ax, pts, color, lw=2.0):
    """Draw a multi-segment polyline through ``pts`` with an arrowhead at end."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, solid_capstyle="round",
            zorder=1)
    straight_arrow(ax, pts[-2][0], pts[-2][1], pts[-1][0], pts[-1][1],
                   color=color, lw=lw)


def arc_arrow(ax, x1, y1, x2, y2, rad, color, lw=2.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}"))


fig, ax = plt.subplots(figsize=(17, 11))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

ax.text(50, 97.5, "auto-psych — automated cognitive-modeling loop",
        ha="center", fontsize=16, fontweight="bold")
ax.text(50, 94.3,
        "Claude Code agents (blue) propose theories, design & run experiments; "
        "programmatic steps (green) collect data and fit/compare PyMC models.",
        ha="center", fontsize=9.5, color="#555")

# ── Legend ──────────────────────────────────────────────────────────────────
for i, (lab, fc, ec) in enumerate([
    ("Claude Code agent (LLM)", AGENT_FC, AGENT_EC),
    ("programmatic step", PROG_FC, PROG_EC),
]):
    ax.add_patch(FancyBboxPatch((2 + i * 22, 91), 2.2, 1.6,
                 boxstyle="round,pad=0.1", facecolor=fc, edgecolor=ec, lw=1.5))
    ax.text(4.6 + i * 22, 91.8, lab, va="center", fontsize=8.5)
ax.plot([46, 49], [91.8, 91.8], color=FLOW, lw=2)
ax.text(49.6, 91.8, "loop / hand-off", va="center", fontsize=8.5)

# ── OUTER LOOP container ─────────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch((2, 56), 96, 32, boxstyle="round,pad=0.4",
             facecolor=OUTER_FC, edgecolor=OUTER_EC, lw=2.2, linestyle="--"))
ax.text(4, 85.6, "OUTER LOOP  ·  experiment loop  (repeat per experiment N)",
        fontsize=12, fontweight="bold", color=OUTER_EC)

centers = [13, 31.5, 50, 68.5, 87]
y_outer = 71
outer_stages = [
    (AGENT_FC, AGENT_EC, "① Theory",
     "Propose PyMC models —\none falsifiable hypothesis\neach (manifest + .py).\nExp 2+: extend prior set"),
    (AGENT_FC, AGENT_EC, "② Design",
     "Generate ~100–300 stimulus\npairs; rank by EIG from\nmodels' prior-predictive.\nKeep top ~20"),
    (AGENT_FC, AGENT_EC, "③ Implement",
     "Build jsPsych experiment\n(index.html + config).\nSkipped in\nno-browser mode"),
    (PROG_FC, PROG_EC, "④ Collect",
     "Gather responses →\nresponses.csv.\nLLM-participant /\nground-truth / prior"),
    (PROG_FC, PROG_EC, "⑤ Model loop",
     "Pool data, run the\ninner loop, export\nbest model + update\nregistry"),
]
for cx, (fc, ec, title, body) in zip(centers, outer_stages):
    draw_box(ax, cx, y_outer, 16, 14, title, body, fc, ec)
for a, b in zip(centers[:-1], centers[1:]):
    straight_arrow(ax, a + 8, y_outer, b - 8, y_outer)

# Outer loop-back: stage 5 -> stage 1, arcing over the top of the boxes.
arc_arrow(ax, 87, y_outer + 7, 13, y_outer + 7, rad=-0.28, color=FLOW)
ax.text(50, 83.2,
        "experiment N+1: carry forward model set + report.md (best model seeds next theory)",
        ha="center", fontsize=9, color=FLOW, style="italic")

# ── INNER LOOP container ─────────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch((2, 8), 96, 38, boxstyle="round,pad=0.4",
             facecolor=INNER_FC, edgecolor=INNER_EC, lw=2.2, linestyle="--"))
ax.text(4, 43.4,
        "INNER LOOP  ·  cognitive-model improvement  (run by stage ⑤ over pooled responses)",
        fontsize=12, fontweight="bold", color=INNER_EC)

y_inner = 28
inner_stages = [
    (PROG_FC, PROG_EC, "Seed model set",
     "Copy this experiment's\ncognitive_models/ as the\n\"zoo\"; drop any seed\nthat can't be fit"),
    (PROG_FC, PROG_EC, "Fit + score",
     "Fit every model by MCMC,\nscore ELPD-LOO →\nposterior over models\n(gentle Occam prior)"),
    (AGENT_FC, AGENT_EC, "Propose candidate",
     "k Claude Code agents read\nexisting hypotheses + fit;\neach writes ONE new /\nrefined PyMC hypothesis"),
    (PROG_FC, PROG_EC, "Admit + re-score",
     "Admit if it loads & has\nfinite logp & states a\nhypothesis; re-fit and\nre-score the whole set"),
    (PROG_FC, PROG_EC, "Export best",
     "Compare (PSIS-LOO),\nwrite report.md +\nmodel_posterior.json,\ncopy best_model.py"),
]
for cx, (fc, ec, title, body) in zip(centers, inner_stages):
    draw_box(ax, cx, y_inner, 16, 14, title, body, fc, ec)
for a, b in zip(centers[:-1], centers[1:]):
    straight_arrow(ax, a + 8, y_inner, b - 8, y_inner)

# Inner repeat loop: "Admit + re-score" -> back to "Fit + score", below boxes.
elbow(ax, [(68.5, 21), (68.5, 13.5), (31.5, 13.5), (31.5, 21)], color=FLOW)
ax.text(50, 12.2, "repeat for each candidate, over max_iterations rounds",
        ha="center", fontsize=9, color=FLOW, style="italic")

# ── Hand-offs between the two loops ──────────────────────────────────────────
# Stage ⑤ invokes the inner loop (down-left into the "Seed" box).
elbow(ax, [(82, y_outer - 7), (82, 51), (8, 51), (8, 35.2)], color=FLOW)
ax.text(45, 52.3, "stage ⑤ invokes inner loop on pooled responses.csv",
        ha="center", fontsize=9, color=FLOW, style="italic")

# Inner loop returns the exported best model up into stage ⑤.
elbow(ax, [(90, 35.2), (90, y_outer - 7)], color=FLOW)
ax.text(94.3, 50, "best_model.py →\ncognitive_models/\ninner_loop_model.py",
        ha="center", va="center", fontsize=8, color=FLOW, style="italic")

out_dir = here("diagrams")
out_dir.mkdir(exist_ok=True)
svg_path = out_dir / "cognitive_modeling_loop.svg"
png_path = out_dir / "cognitive_modeling_loop.png"
fig.savefig(svg_path, bbox_inches="tight")
fig.savefig(png_path, dpi=200, bbox_inches="tight")
print(f"Wrote {svg_path}")
print(f"Wrote {png_path}")
