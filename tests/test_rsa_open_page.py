"""End to end: build a paradigm's page, answer it in Chromium, read its data back as the link's.

Skipped where Playwright or the Chromium build is missing, or the jsPsych CDN is unreachable
(every other request is aborted, and the test fails if the page made one).
"""

import json
import urllib.error

import numpy as np
import pytest

from tests.rsa_experiment_browser import CHROMIUM, chromium_available, fetch_cdn, install_routes
from tests.test_rsa_open_screens import KINDS, paradigm_of, units_of

pytestmark = pytest.mark.skipif(not chromium_available(), reason=f"Playwright or Chromium ({CHROMIUM}) missing")


def test_the_page_carries_its_lists_and_a_consent_placeholder():
    from src.rsa_open.page import CONSENT_PLACEHOLDER_MARKER, render
    from src.rsa_open.screens import trial_lists

    p = paradigm_of("choice")
    lists = trial_lists(p, units_of(p), seed=0, n_lists=2)
    page = render(lists)
    assert CONSENT_PLACEHOLDER_MARKER in page and "{{" not in page.split("<script>")[1].split("const LISTS")[0]
    assert render(lists, consent_html=None).count(CONSENT_PLACEHOLDER_MARKER) == 1  # the CSS rule only
    with pytest.raises(ValueError, match="schema"):
        render({"lists": []})


@pytest.fixture(scope="module")
def cdn():
    try:
        return fetch_cdn()
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        pytest.skip(f"jsPsych CDN unreachable: {e}")


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        b = pw.chromium.launch(executable_path=str(CHROMIUM))
        yield b
        b.close()


def answer_everything(page, timeout_ms=15000):
    """Answer every screen; get the first comprehension check wrong once, then right."""
    checks_answered = 0
    while not page.evaluate("window.__roDone === true"):
        page.wait_for_function("window.__roDone === true || document.querySelector("
                               "'.ro-trial, .ro-option, .ro-next') !== null", timeout=timeout_ms)
        if page.evaluate("window.__roDone === true"):
            break
        if page.locator(".ro-trial").count():
            if page.locator("button.ro-item").count():
                page.locator("button.ro-item").first.click()
            elif page.locator(".ro-rate").count():
                page.locator(".ro-rate").last.click()
            elif page.locator(".ro-slider").count():
                assert page.locator(".ro-continue").is_disabled()  # untouched, no answer
                assert page.locator(".ro-slider.ro-unset").count() == 1  # no handle until placed
                hidden = page.locator(".ro-slider").screenshot()
                page.evaluate("document.querySelector('.ro-slider').classList.remove('ro-unset')")
                assert page.locator(".ro-slider").screenshot() != hidden  # the same value, its handle now drawn
                page.evaluate("document.querySelector('.ro-slider').classList.add('ro-unset')")
                assert page.locator(".ro-continue").is_disabled()
                page.locator(".ro-slider").fill("73")
                assert page.locator(".ro-slider.ro-unset").count() == 0
                page.locator(".ro-continue").click()
            elif page.locator(".ro-bet").count():
                total = int(page.locator(".ro-total").inner_text().split(" of ")[1].split()[0])
                page.locator(".ro-bet").first.fill(str(total - 1))
                assert page.locator(".ro-continue").is_disabled()  # points must add up
                page.locator(".ro-bet").last.fill("1")
                page.locator(".ro-continue").click()
            elif page.locator(".ro-word-btn").count():
                page.locator(".ro-word-btn").first.click()
            else:
                page.locator(".ro-answer").first.click()
        elif page.locator(".ro-option").count():
            page.locator(".ro-option").nth(1 if checks_answered == 0 else 0).click()
            checks_answered += 1
        else:
            page.locator(".ro-next").first.click()
        page.wait_for_timeout(30)
    return json.loads(page.evaluate("window.__roData"))


@pytest.mark.parametrize("kind", KINDS)
def test_answering_the_page_gives_the_link_s_data(kind, browser, cdn, tmp_path):
    from src.rsa_open.fit import OpenData
    from src.rsa_open.page import render
    from src.rsa_open.paradigm import paradigm_likelihood
    from src.rsa_open.screens import page_rows, scenes_and_responses, trial_lists

    p = paradigm_of(kind, trials_per_participant=4, catch_trials=1)
    lists = trial_lists(p, units_of(p), seed=5, n_lists=3)
    path = tmp_path / "page.html"
    path.write_text(render(lists, preview=True))
    context = browser.new_context(viewport={"width": 1100, "height": 900})
    page = context.new_page()
    blocked, errors = [], []
    install_routes(page, cdn, blocked)
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"file://{path}?list=1")
    data = answer_everything(page)
    context.close()
    assert not blocked and not errors, (blocked, errors)
    # The comprehension check was failed once: the instructions came back, and it was asked again.
    assert [d["task"] for d in data].count("instructions") == 2
    assert [d["correct"] for d in data if d["task"] == "comprehension"] == [False, True]
    rows = page_rows(data)
    assert [r["trial_number"] for r in rows] == list(range(6))
    assert all(r["list_index"] == 1 and r["paradigm_sha256"] == p.sha256 for r in rows)
    expected = [t for t in lists["lists"][1]["trials"]]
    for row, trial in zip(rows, expected):
        assert row["phase"] == trial["phase"] and row["objects"] == trial["objects"]
    scenes, responses = scenes_and_responses(rows)
    data_ = OpenData.build(scenes, responses, paradigm_likelihood(p))
    assert data_.n_trials == 4
    if kind == "choice":  # the first screen position, recorded as its scene index
        assert [r["response"]["choice"] for r in rows] == [t["display_order"][0] for t in expected]
    if kind == "slider":
        assert {r["response"]["value"] for r in rows} == {73}
    if kind == "bet":
        for r, t in zip(rows, expected):
            points = np.zeros(len(t["objects"]), int)
            points[t["display_order"][0]] += p.response.total_points - 1
            points[t["display_order"][-1]] += 1
            assert r["response"]["points"] == points.tolist()


def test_a_two_measure_trial_asks_the_second_question_on_the_same_screen(browser, cdn, tmp_path):
    """The second question comes once the first is answered; the page records both, the second
    under "second" with its own probed object (src/rsa_open/links.py PairLikelihood)."""
    from src.rsa_open.fit import OpenData
    from src.rsa_open.page import render
    from src.rsa_open.paradigm import paradigm_likelihood
    from src.rsa_open.screens import page_rows, scenes_and_responses, trial_lists
    from tests.test_rsa_open_two_measures import pair_paradigm

    p = pair_paradigm("choice", "rating", trials_per_participant=4, catch_trials=1)
    lists = trial_lists(p, units_of(p), seed=5, n_lists=3)
    path = tmp_path / "page.html"
    path.write_text(render(lists, preview=True))
    context = browser.new_context(viewport={"width": 1100, "height": 900})
    page = context.new_page()
    blocked, errors = [], []
    install_routes(page, cdn, blocked)
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"file://{path}?list=1")
    data = answer_everything(page)
    context.close()
    assert not blocked and not errors, (blocked, errors)
    rows = page_rows(data)
    expected = lists["lists"][1]["trials"]
    assert len(rows) == len(expected) == 6
    for row, trial in zip(rows, expected):
        assert row["response"]["choice"] == trial["display_order"][0]  # the first object on screen
        assert row["response"]["second"] == {"probe": trial["second_probe"], "rating": 5}  # the top of the scale
        assert row["second_probe"] == trial["second_probe"]
    scenes, responses = scenes_and_responses(rows)
    assert OpenData.build(scenes, responses, paradigm_likelihood(p)).n_trials == 4
