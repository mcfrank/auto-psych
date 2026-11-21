"""The domains' stimulus icons, beside each panel's title (``add_title_icons``). Each draws into an axes
``STIMULI`` gives the width of (its height is 1)."""
import matplotlib

matplotlib.use("Agg")
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle  # noqa: E402

from scripts.paper.figures.style import INK, INK_2  # noqa: E402

GREY = "#bdbdbd"


def _face(ax, x, y, r, glasses=False, hat=False, color=GREY):
    """A face: a disc, with glasses (two rims and a bridge) and/or a hat."""
    ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor="none"))
    if glasses:
        for dx in (-0.42 * r, 0.42 * r):
            ax.add_patch(Circle((x + dx, y + 0.12 * r), 0.3 * r, facecolor="white", edgecolor=INK, lw=0.9))
        ax.plot([x - 0.12 * r, x + 0.12 * r], [y + 0.12 * r] * 2, color=INK, lw=0.9)
    if hat:
        ax.add_patch(Rectangle((x - 1.05 * r, y + 0.82 * r), 2.1 * r, 0.16 * r, facecolor=INK, edgecolor="none"))
        ax.add_patch(Rectangle((x - 0.6 * r, y + 0.95 * r), 1.2 * r, 0.6 * r, facecolor=INK, edgecolor="none"))




def _strip(ax, width):
    """An axes [0, width] x [0, 1], equal aspect, no frame."""
    ax.set_xlim(0, width)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")


def randomness_stimulus(ax):
    """The two coin sequences, heads filled."""
    _strip(ax, 2.6)
    for y, seq in ((0.72, "HTHHTHTT"), (0.28, "HHHHTTTT")):
        for i, c in enumerate(seq):
            ax.add_patch(Circle((0.2 + i * 0.31, y), 0.12, facecolor=INK_2 if c == "H" else "white",
                                edgecolor=INK_2, lw=0.6))


def reference_stimulus(ax):
    """The three faces."""
    _strip(ax, 1.6)
    for x, kw in ((0.25, {}), (0.8, {"glasses": True}), (1.35, {"glasses": True, "hat": True})):
        _face(ax, x, 0.4, 0.21, color=GREY, **kw)


def number_stimulus(ax):
    """The example cards."""
    _strip(ax, 2.3)
    for k, n in enumerate(("16", "8", "2", "64")):
        x = 0.27 + k * 0.58
        ax.add_patch(FancyBboxPatch((x - 0.25, 0.25), 0.5, 0.5, boxstyle="round,pad=0,rounding_size=0.08",
                                    facecolor="white", edgecolor=INK_2, lw=0.6))
        ax.text(x, 0.5, n, ha="center", va="center", fontsize=4.5, color=INK)


STIMULI = {"subjective_randomness": (randomness_stimulus, 2.6), "rsa": (reference_stimulus, 1.6),
           "number_game": (number_stimulus, 2.3)}  # each icon and its width (its height is 1)


def add_title_icons(fig, axes, domains) -> None:
    """Each domain's stimulus icon just left of its axes' (centred) title."""
    fig.canvas.draw()
    to_fig = fig.transFigure.inverted()
    for ax, d in zip(axes, domains):
        box = ax.title.get_window_extent()
        (x0, y0), (x1, y1) = to_fig.transform((box.x0, box.y0)), to_fig.transform((box.x1, box.y1))
        draw, aspect = STIMULI[d]
        h = 1.9 * (y1 - y0)
        w = h * aspect * fig.get_figheight() / fig.get_figwidth()
        draw(fig.add_axes((x0 - w - 0.008, (y0 + y1) / 2 - h / 2, w, h)))

