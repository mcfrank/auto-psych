"""Figure 1 (``fig1_loops``): the framework's two loops (a) and the three domains (b).

The figure is drawn by hand, not from data: its source is ``fig1.html`` (an inline SVG whose user unit is
the PDF point, styled by the CSS in its head). This module prints that page to PDF with Playwright's
Chromium, as the original figure was printed by headless Chrome, so the text stays live text in Avenir
and Menlo (taken from the system, like ``style.py``'s Avenir). The PNG is a 600 dpi screenshot of the
same page.

Needs Playwright's Chromium once per machine: ``uv run playwright install chromium``.
"""

from __future__ import annotations

from pathlib import Path

from scripts.paper.figures.style import DPI

SOURCE = Path(__file__).with_name("fig1.html")
NAME = "fig1_loops"
WIDTH_PT, HEIGHT_PT = 505.92, 214.08  # the page, as in fig1.html's @page (the original figure's size)
CSS_PX_PER_PT = 96 / 72


def draw(out_dir: Path) -> None:
    """Write ``fig1_loops.pdf`` (vector, live text) and ``fig1_loops.png`` (600 dpi) to ``out_dir``."""
    from playwright.sync_api import sync_playwright

    out_dir.mkdir(parents=True, exist_ok=True)
    width_px, height_px = WIDTH_PT * CSS_PX_PER_PT, HEIGHT_PT * CSS_PX_PER_PT
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": round(width_px), "height": round(height_px)},
                                device_scale_factor=DPI / 96)
        page.goto(SOURCE.resolve().as_uri())
        page.evaluate("document.fonts.ready")
        page.pdf(path=str(out_dir / f"{NAME}.pdf"), prefer_css_page_size=True, print_background=True)
        page.screenshot(path=str(out_dir / f"{NAME}.png"),
                        clip={"x": 0, "y": 0, "width": width_px, "height": height_px})
        browser.close()
