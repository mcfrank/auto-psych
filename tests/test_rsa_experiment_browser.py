"""End to end: build the reference-game page, click through it in Chromium, convert its data.

Skipped where Playwright or the Chromium build is missing, or the jsPsych CDN
is unreachable (the page itself needs nothing else from the network: every
other request is aborted and the test fails if there was one).
"""

import json
import urllib.error

import pytest

from src.rsa.dataset import context_from_row
from src.rsa.experiment.build import CONSENT_PLACEHOLDER_MARKER, build_site
from src.rsa.experiment.convert import convert
from src.rsa.experiment.design import EXPERIMENT_ASSETS_DIR, Design, trial_lists
from tests.rsa_experiment_browser import CHROMIUM, chromium_available, click_through, fetch_cdn, install_routes

pytestmark = pytest.mark.skipif(not chromium_available(), reason=f"Playwright or Chromium ({CHROMIUM}) missing")

KEYBOARD_TRIAL = 2


@pytest.fixture(scope="module")
def cdn():
    try:
        return fetch_cdn()
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        pytest.skip(f"jsPsych CDN unreachable: {e}")


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    doc = trial_lists(Design.load(EXPERIMENT_ASSETS_DIR / "demo_design.json"), seed=41, n_lists=3, n_catch=2)
    index = build_site(doc, tmp_path_factory.mktemp("site"))
    return doc, index


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=str(CHROMIUM))
        yield b
        b.close()


def _run(browser, cdn, url, *, width=1280, height=800, on_choice_screen=None):
    context = browser.new_context(viewport={"width": width, "height": height})
    page = context.new_page()
    blocked, errors = [], []
    page.on("pageerror", lambda e: errors.append(str(e)))
    install_routes(page, cdn, blocked)
    page.goto(url)
    consent_seen = []
    page.wait_for_selector(".rsa-next")
    page.locator(".rsa-next").first.click()  # welcome
    page.wait_for_selector(".rsa-next")
    consent_seen.append(CONSENT_PLACEHOLDER_MARKER in page.content())
    n = click_through(page, keyboard_trial=KEYBOARD_TRIAL, on_choice_screen=on_choice_screen)
    data = page.evaluate("window.__rsaData")
    end_text = page.locator("#rsa-end").inner_text()
    context.close()
    assert not blocked, f"the page requested {blocked}"
    assert not errors, errors
    assert consent_seen == [True]
    return n, data, end_text


def test_a_scripted_participant_completes_a_list_and_its_data_convert(browser, cdn, site):
    doc, index = site
    list_index = 1
    lst = doc["lists"][list_index]
    screens = []

    def on_choice_screen(page):
        refs = page.locator(".rsa-ref")
        alts = page.locator(".rsa-ref .rsa-stack img:first-child").evaluate_all("els => els.map(e => e.alt)")
        tags = refs.evaluate_all("els => els.map(e => e.tagName)")
        screens.append((page.locator(".rsa-prompt").inner_text(), alts, tags))

    n, data, end_text = _run(
        browser, cdn, index.as_uri() + f"?list={list_index}&participant_id=SIM1", on_choice_screen=on_choice_screen
    )
    assert n == len(lst["trials"])
    assert "Thank you" in end_text

    # What was on screen is what the list says, with alt text naming each object's features.
    for (prompt, alts, tags), trial in zip(screens, lst["trials"]):
        assert tags == ["BUTTON"] * len(trial["objects"])
        assert alts == [cell["alt"] for cell in trial["screen"]]
        assert f"Click on the {trial['item']}" in prompt
        assert (trial["word"] in prompt) if trial["word"] else ("mumble" in prompt)

    records = json.loads(data)
    assert {r["list_index"] for r in records} == {list_index}
    assert {r["list_assignment"] for r in records} == {"url"}
    assert {r["participant_id"] for r in records} == {"SIM1"}

    frame = convert(data, participant_id="sim1", experiment="rsa_browser_test")
    tests = lst["trials"][1:]
    assert len(frame) == len(tests)
    for (_, row), trial in zip(frame.iterrows(), tests):
        ctx = context_from_row(row)
        assert ctx.objects == tuple(tuple(r) for r in trial["objects"])
        assert list(ctx.feature_names) == trial["feature_names"]
        assert ctx.utterance == trial["utterance"]
        # The scripted participant clicks the leftmost referent, except on one
        # trial where it picks the second one with the keyboard.
        position = 1 if trial["trial_number"] == KEYBOARD_TRIAL else 0
        assert row["choice"] == trial["display_order"][position]
        cov = json.loads(row["covariates"])
        assert cov["choice_position"] == position and cov["rt"] > 0


def test_without_a_list_parameter_a_random_list_runs_on_a_phone(browser, cdn, site):
    doc, index = site
    overflow = []

    def on_choice_screen(page):
        if page.evaluate("document.documentElement.scrollWidth > window.innerWidth"):
            overflow.append(page.locator(".rsa-prompt").inner_text())

    _, data, _ = _run(browser, cdn, index.as_uri(), width=390, height=844, on_choice_screen=on_choice_screen)
    assert not overflow, "the page scrolls horizontally at phone width"
    records = json.loads(data)
    assert {r["list_assignment"] for r in records} == {"random"}
    (k,) = {r["list_index"] for r in records}
    assert 0 <= k < len(doc["lists"])
    assert {r["participant_id"] for r in records} == {None}
    frame = convert(data, participant_id="sim2", experiment="rsa_browser_test")
    assert len(frame) == len(doc["lists"][k]["trials"]) - 1


def test_an_invalid_list_parameter_stops_the_page(browser, cdn, site):
    _, index = site
    context = browser.new_context()
    page = context.new_page()
    install_routes(page, cdn, [])
    page.goto(index.as_uri() + "?list=99")
    assert "Invalid list parameter" in page.locator(".rsa-error").inner_text()
    context.close()


def test_the_page_keeps_its_styles_when_a_host_puts_it_inside_body(browser, cdn, tmp_path):
    """An artifact host wraps the page in its own skeleton, so the page's
    <style> sits in <body>. jsPsych must not run on <body> (it clears it):
    the first published preview lost every style that way, and each object's
    feature overlays rendered beside it instead of on top."""
    from src.rsa.experiment.build import build_single_file

    doc = trial_lists(Design.load(EXPERIMENT_ASSETS_DIR / "demo_design.json"), seed=41, n_lists=1, n_catch=2)
    single = build_single_file(doc, tmp_path / "preview.html", preview=True)
    hosted = tmp_path / "hosted.html"
    hosted.write_text("<!doctype html><html><head><meta charset='utf-8'></head><body>"
                      + single.read_text(encoding="utf-8") + "</body></html>", encoding="utf-8")
    context = browser.new_context(viewport={"width": 1280, "height": 800})
    page = context.new_page()
    install_routes(page, cdn, [])
    page.goto(hosted.as_uri())
    while not page.locator(".rsa-ref").count():
        page.wait_for_selector(".rsa-next, .rsa-ref")
        if page.locator(".rsa-ref").count():
            break
        page.locator(".rsa-next").first.click()
    page.wait_for_selector(".rsa-ref")
    # Every image of a referent is drawn in the same square: the overlays stack.
    rects = page.locator(".rsa-ref").first.locator("img").evaluate_all(
        "els => els.map(e => { const r = e.getBoundingClientRect(); return [r.x, r.y, r.width, r.height]; })")
    assert len(rects) >= 1 and all(r == rects[0] for r in rects), rects
    assert page.locator(".rsa-ref").first.evaluate("e => getComputedStyle(e).borderRadius") != "0px"
    context.close()
