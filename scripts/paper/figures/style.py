"""The paper figures' house style (rational-reps plot_utils): Avenir, its palette, open axes.

Avenir is a commercial font and is not in the repo: ``setup_style`` takes it from the first of
``AVENIR_PATHS`` that exists (macOS ships it), else the closest sans serif installed
(``FALLBACK_FAMILIES``), saying so.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

DOMAINS = ["subjective_randomness", "rsa", "number_game"]
TITLES = {"subjective_randomness": "Subjective randomness", "rsa": "Reference games",
          "number_game": "Number game"}
MOCK_MARK = " (MOCK DATA)"  # appended to a mock domain's titles

INK, INK_2 = "#262626", "#595959"
TITLE_FONT = {"family": "Avenir", "weight": "medium"}  # family: the font setup_style found
FULL_WIDTH_IN = 7.2  # a two-column paper's text width

AVENIR_PATHS = [Path.home() / ".fonts" / "Avenir.ttc", Path("/System/Library/Fonts/Avenir.ttc"),
                Path("/Library/Fonts/Avenir.ttc")]
FACES = {"Avenir Book": 0, "Avenir Book Oblique": 1, "Avenir Medium": 8}  # Avenir.ttc's face indices
# Avenir's nearest relatives first; DejaVu Sans ships with matplotlib, so one is always found.
FALLBACK_FAMILIES = ["Avenir Next", "Nunito Sans", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
FONT_CACHE = Path(matplotlib.get_cachedir()) / "avenir"


def panel_title(domain: str, *, mock_domains: List[str]) -> str:
    return f"{TITLES[domain]}{MOCK_MARK}" if domain in mock_domains else TITLES[domain]


def _register_avenir(ttc: Path) -> None:
    """matplotlib reads only a TTC's first face, so each face is written out once as its own TTF."""
    from fontTools.ttLib import TTCollection

    collection = None
    for name, index in FACES.items():
        path = FONT_CACHE / f"{name.replace(' ', '')}.ttf"
        if not path.exists():
            collection = collection or TTCollection(str(ttc))
            face = collection.fonts[index]
            if face["name"].getDebugName(4) != name:
                raise RuntimeError(f"{ttc}: face {index} is {face['name'].getDebugName(4)!r}, not {name!r}")
            path.parent.mkdir(parents=True, exist_ok=True)
            face.save(str(path))
        fm.fontManager.addfont(str(path))


def setup_style() -> None:
    """Avenir Book for text, Medium for titles, Book Oblique for italics (where Avenir is installed).
    Changes ``plt.rcParams``: call it inside ``plt.rc_context()``."""
    found = [p for p in AVENIR_PATHS if p.exists()]
    if found:
        _register_avenir(found[0])
        family = ["Avenir"]
    else:
        installed = {f.name for f in fm.fontManager.ttflist}
        family = [f for f in FALLBACK_FAMILIES if f in installed]
        print(f"Avenir not found ({', '.join(map(str, AVENIR_PATHS))}): the figures use {family[0]}", flush=True)
    plt.rcParams.update({
        "font.family": family, "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "cm",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.edgecolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    })
    TITLE_FONT["family"] = family[0]


def save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
