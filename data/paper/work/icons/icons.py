"""Draft task icons (throwaway; Figure 1?): each draws a task's scene into an axes in [0, 1] x [0, 1].
The figures' title icons are scripts/paper/figures/icons.py.

Run: uv run python -m data.paper.work.icons.icons
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle  # noqa: E402
from pyprojroot import here  # noqa: E402

from scripts.paper.figures.icons import _face  # noqa: E402
from scripts.paper.figures.style import INK, INK_2, setup_style  # noqa: E402

OUT = here() / "data" / "paper" / "work" / "icons"
DARK, MID, LIGHT = "#08306b", "#4292c6", "#c6dbef"  # the figures' blues
GREY = "#bdbdbd"
SPEAKER = "#d9d9d9"  # the speaker's fill; the listener is white, so the nested thoughts read


def _blank(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")


def _coin_row(ax, x0, y, seq, r, step):
    for i, c in enumerate(seq):
        ax.add_patch(Circle((x0 + i * step, y), r, facecolor=DARK if c == "H" else "white", edgecolor=DARK,
                            lw=0.9, zorder=3))


def randomness_icon(ax):
    """Which sequence came from a fair coin? (Falk & Konold, 1997; Griffiths & Tenenbaum, 2003):
    a person weighs two sequences against a fair coin and picks the alternating one."""
    from matplotlib.patches import Wedge
    _blank(ax)
    r, step, x0 = 0.03, 0.08, 0.36
    _coin_row(ax, x0, 0.34, "HTHHTHTT", r, step)
    _coin_row(ax, x0, 0.18, "HHHHTTTT", r, step)
    pick = FancyBboxPatch((x0 - 0.058, 0.34 - 0.058), 7 * step + 0.116, 0.116,
                          boxstyle="round,pad=0,rounding_size=0.06", facecolor="none", edgecolor=MID, lw=1.1, zorder=4)
    ax.add_patch(pick)
    _person(ax, 0.11, 0.15, 0.07)
    # the thought: a fair coin, half heads and half tails
    _cloud(ax, 0.42, 0.76, 0.42, 0.28, trail=(0.12, 0.24), z=1)
    cx, cy, cr = 0.42, 0.76, 0.065
    ax.add_patch(Wedge((cx, cy), cr, 90, 270, facecolor=DARK, edgecolor=DARK, lw=0.9, zorder=3))
    ax.add_patch(Wedge((cx, cy), cr, 270, 90, facecolor="white", edgecolor=DARK, lw=0.9, zorder=3))


def _cloud(ax, cx, cy, w, h, trail, z=1, lw=0.8):
    """A thought cloud: an outline of overlapping bumps around an ellipse at (cx, cy), with a
    trail of two shrinking circles toward ``trail`` (the thinker's head). ``z`` stacks clouds
    drawn inside clouds."""
    import numpy as np
    from matplotlib.patches import Ellipse
    br = 0.2 * h  # bump radius, so the bumps overlap at any size
    n = max(7, int(np.pi * (w + h) / 2 / (1.45 * br)))
    bumps = [(cx + 0.5 * w * np.cos(t), cy + 0.5 * h * np.sin(t)) for t in np.linspace(0, 2 * np.pi, n + 1)[:-1]]
    for c in bumps:  # outlined, overlapping bumps ...
        ax.add_patch(Circle(c, br, facecolor="white", edgecolor=INK_2, lw=lw, zorder=z))
    for c in bumps:  # ... whose inner arcs a second, unoutlined pass covers, leaving the outline
        ax.add_patch(Circle(c, br * 0.93, facecolor="white", edgecolor="none", zorder=z + 0.1))
    ax.add_patch(Ellipse((cx, cy), w, h, facecolor="white", edgecolor="none", zorder=z + 0.1))
    (x0, y0) = trail
    x1, y1 = cx + 0.5 * w * np.cos(np.arctan2(y0 - cy, x0 - cx)), cy + 0.5 * h * np.sin(np.arctan2(y0 - cy, x0 - cx))
    for k, f in enumerate((0.3, 0.62)):
        ax.add_patch(Circle((x0 + f * (x1 - x0), y0 + f * (y1 - y0)), (0.3 + 0.25 * k) * br, facecolor="white",
                            edgecolor=INK_2, lw=lw, zorder=z))


def _person(ax, x, y, r, z=5, lw=0.9, fill="white"):
    """A minimal person: an outlined head over outlined shoulders, so people read apart from
    the filled faces the game is about."""
    import numpy as np
    from matplotlib.patches import Wedge
    ax.add_patch(Wedge((x, y - 1.05 * r), 1.25 * r, 0, 180, facecolor=fill, edgecolor=INK_2, lw=lw, zorder=z))
    ax.add_patch(Circle((x, y), 0.62 * r, facecolor=fill, edgecolor=INK_2, lw=lw, zorder=z + 0.1))


def reference_icon(ax):
    """Rational speech acts (Goodman & Frank, 2016, Fig. 1): the listener reasons about a speaker
    who reasons about a literal listener (nested thoughts); on hearing "glasses" the listener
    picks the face with glasses only, since a speaker meaning the one with a hat too would have
    said "hat"."""
    _blank(ax)
    # the scene: listener, the three faces, speaker with the utterance
    _person(ax, 0.11, 0.15, 0.07)
    _person(ax, 0.89, 0.15, 0.07, fill=SPEAKER)
    y, r = 0.13, 0.062
    for x, kw in ((0.32, {}), (0.5, {"glasses": True}), (0.68, {"glasses": True, "hat": True})):
        _face(ax, x, y, r, color=GREY, **kw)
    # the answer, the face with glasses only: a rounded box, as the other icons mark theirs
    m = r + 0.022
    ax.add_patch(FancyBboxPatch((0.5 - m, y - m), 2 * m, 2 * m, boxstyle="round,pad=0,rounding_size=0.03",
                                facecolor="none", edgecolor=MID, lw=1.1, zorder=6))
    bubble = FancyBboxPatch((0.64, 0.33), 0.3, 0.1, boxstyle="round,pad=0.012,rounding_size=0.035",
                            facecolor="white", edgecolor=INK_2, lw=0.8, zorder=3)
    ax.add_patch(bubble)
    ax.add_patch(Polygon([[0.83, 0.325], [0.89, 0.325], [0.89, 0.25]], closed=True, facecolor="white",
                         edgecolor="none", zorder=4))
    ax.plot([0.83, 0.89, 0.89], [0.325, 0.25, 0.325], color=INK_2, lw=0.8, zorder=5, solid_joinstyle="round")
    ax.text(0.79, 0.38, "“glasses”", ha="center", va="center", fontsize=5.5, color=INK, zorder=6)
    # the listener thinks of the speaker, who thinks of a (literal) listener
    _cloud(ax, 0.42, 0.7, 0.56, 0.36, trail=(0.12, 0.24), z=1)
    _person(ax, 0.27, 0.66, 0.06, z=3, lw=0.8, fill=SPEAKER)
    _cloud(ax, 0.5, 0.75, 0.26, 0.17, trail=(0.29, 0.7), z=3, lw=0.7)
    _person(ax, 0.5, 0.76, 0.04, z=5, lw=0.7)



def reference_icon_compact(ax):
    """`reference_icon` with the faces inside the listener's thought: the listener hears "glasses"
    and, reasoning about a speaker who reasons about a literal listener, picks the face with
    glasses only."""
    _blank(ax)
    _person(ax, 0.11, 0.15, 0.07)
    _person(ax, 0.89, 0.15, 0.07)
    bubble = FancyBboxPatch((0.5, 0.11), 0.28, 0.1, boxstyle="round,pad=0.012,rounding_size=0.035",
                            facecolor="white", edgecolor=INK_2, lw=0.8, zorder=3)
    ax.add_patch(bubble)
    ax.add_patch(Polygon([[0.78, 0.13], [0.78, 0.19], [0.82, 0.16]], closed=True, facecolor="white",
                         edgecolor="none", zorder=4))
    ax.plot([0.78, 0.82, 0.78], [0.13, 0.16, 0.19], color=INK_2, lw=0.8, zorder=5, solid_joinstyle="round")
    ax.text(0.64, 0.16, "“glasses”", ha="center", va="center", fontsize=5.5, color=INK, zorder=6)
    # the listener's thought: the speaker, who thinks of a literal listener, and the faces
    _cloud(ax, 0.5, 0.63, 0.66, 0.5, trail=(0.13, 0.24), z=1)
    _person(ax, 0.32, 0.74, 0.05, z=3, lw=0.8)
    _cloud(ax, 0.56, 0.8, 0.22, 0.13, trail=(0.34, 0.77), z=3, lw=0.7)
    _person(ax, 0.56, 0.8, 0.032, z=5, lw=0.7)
    y, r = 0.52, 0.055
    for x, kw in ((0.36, {"color": GREY}), (0.5, {"glasses": True}), (0.64, {"glasses": True, "hat": True, "color": GREY})):
        _face(ax, x, y, r, color=MID, **kw)
    for art in ax.patches[-8:]:
        art.set_zorder(max(art.get_zorder(), 3))
    for line in ax.lines[-2:]:
        line.set_zorder(4)


def _card(ax, x, y, text, edge=INK_2, fill="white", color=INK, w=0.11, h=0.11, size=5.5):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.02",
                                facecolor=fill, edgecolor=edge, lw=0.8, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=color, zorder=4)


def number_icon(ax):
    """From examples to a program (Tenenbaum, 2000): seeing 16, 8, 2 and 64, a person infers the
    rule (powers of two, 2ⁿ, beats even numbers, 2n) and so says yes to 32."""
    _blank(ax)
    for k, n in enumerate((16, 8, 2, 64)):
        _card(ax, 0.3 + k * 0.13, 0.16, str(n), w=0.115)
    _card(ax, 0.88, 0.16, "32", edge=MID, color=DARK)
    _person(ax, 0.1, 0.15, 0.07)
    # the thought: candidate programs and how much each is believed
    _cloud(ax, 0.47, 0.67, 0.62, 0.4, trail=(0.11, 0.25), z=1)
    rules = (("2ⁿ", 0.9, DARK, INK), ("2n", 0.1, LIGHT, INK_2))  # powers of two, even numbers
    for k, (rule, prob, bar, ink) in enumerate(rules):
        y = 0.74 - k * 0.12
        base = ax.text(0.29, y, "2" if rule == "2ⁿ" else rule, ha="left", va="center", fontsize=7, color=ink, zorder=4)
        if rule == "2ⁿ":  # the exponent, set by hand: a small raised n right after the 2
            ax.annotate("n", xy=(1, 0.75), xycoords=base, ha="left", va="center", fontsize=5, color=ink, zorder=4)
        ax.add_patch(Rectangle((0.47, y - 0.035), 0.26 * prob, 0.06, facecolor=bar, edgecolor="none", zorder=4))


# --- stimulus-only versions, for panel titles: no people, thoughts or answer, in ink -----------


def main():
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(4.5, 1.4))
    for ax, draw, title in zip(axes, (randomness_icon, reference_icon, number_icon),
                               ("Subjective randomness", "Reference games", "Number game")):
        draw(ax)
        ax.set_title(title, fontsize=7.5, pad=2)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"icons.{ext}", dpi=400, bbox_inches="tight")


if __name__ == "__main__":
    main()
