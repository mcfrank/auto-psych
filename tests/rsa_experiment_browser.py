"""Driving the reference-game page in headless Chromium (Playwright).

The page loads jsPsych from unpkg. Chromium here does not trust the agent
proxy's CA, so the CDN requests are answered from a local copy fetched with
Python (which does), and every other non-file request is aborted and recorded:
the page must need nothing else from the network.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional

CHROMIUM = Path("/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
CDN_URLS = (
    "https://unpkg.com/jspsych@8.2.3",
    "https://unpkg.com/@jspsych/plugin-html-button-response@2.1.0",
)


def chromium_available() -> bool:
    try:
        import playwright.sync_api  # noqa: F401
    except ImportError:
        return False
    return CHROMIUM.exists()


def fetch_cdn() -> Dict[str, tuple]:
    """url -> (content type, bytes) for the page's CDN files (raises when unreachable)."""
    out = {}
    for url in CDN_URLS:
        with urllib.request.urlopen(url, timeout=30) as r:
            out[url] = (r.headers.get("Content-Type", "application/javascript"), r.read())
    return out


def install_routes(page, cdn: Dict[str, tuple], blocked: List[str]) -> None:
    def handle(route):
        url = route.request.url
        if url.startswith("file://") or url.startswith("data:"):
            return route.continue_()
        if url in cdn:
            content_type, body = cdn[url]
            return route.fulfill(status=200, body=body, headers={"Content-Type": content_type})
        blocked.append(url)
        return route.abort()

    page.route("**/*", handle)


def click_through(
    page,
    *,
    keyboard_trial: Optional[int] = None,
    on_choice_screen: Optional[Callable] = None,
    timeout_ms: int = 15000,
) -> int:
    """Answer every screen: the first referent button on each choice screen.

    On the test trial ``keyboard_trial`` the second referent is chosen with
    the keyboard (focus + Enter) instead. ``on_choice_screen(page)`` runs on
    every choice screen before the answer. Returns the number of choices made.
    """
    choices = 0
    test_trials_seen = 0
    while not page.evaluate("window.__rsaDone === true"):
        page.wait_for_function(
            "window.__rsaDone === true || document.querySelector('.rsa-ref, .rsa-next') !== null",
            timeout=timeout_ms,
        )
        if page.evaluate("window.__rsaDone === true"):
            break
        refs = page.locator(".rsa-ref")
        if refs.count():
            is_practice = page.locator(".rsa-practice-tag").count() > 0
            if on_choice_screen is not None:
                on_choice_screen(page)
            if not is_practice and test_trials_seen == keyboard_trial:
                refs.nth(1).focus()
                page.keyboard.press("Enter")
            else:
                refs.first.click()
            if not is_practice:
                test_trials_seen += 1
            choices += 1
        else:
            page.locator(".rsa-next").first.click()
        # Wait for the screen to change (jsPsych clears the display at once).
        page.wait_for_timeout(50)
    return choices
