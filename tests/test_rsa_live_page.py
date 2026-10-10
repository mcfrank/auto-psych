"""The live page in Chromium: the IRB consent gate, a server-assigned list, the post to /submit
and the redirect to Prolific, against stand-ins for the Cloud Functions (no network).

The page is staged as main's deployment stages it (`ensure_consent_gate`), and
the deployment's generic submit bridge must find nothing to add.
"""

import json
import urllib.error
from pathlib import Path

import pytest

from src.pipelines.outer_loop.deployment.firebase import (
    CONSENT_GATE_MARKER,
    ensure_consent_gate,
    ensure_submit_bridge,
    load_consent_html,
)
from src.rsa.experiment.convert import convert
from src.rsa.experiment.design import EXPERIMENT_ASSETS_DIR, Design, trial_lists
from src.rsa.live.site import LIVE_MARKER, build_live_site
from tests.rsa_experiment_browser import CHROMIUM, chromium_available, fetch_cdn

pytestmark = pytest.mark.skipif(not chromium_available(), reason=f"Playwright or Chromium ({CHROMIUM}) missing")

ORIGIN = "https://live.test"
REDIRECT = f"{ORIGIN}/done?cc=RSA123"


@pytest.fixture(scope="module")
def cdn():
    try:
        return fetch_cdn()
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        pytest.skip(f"jsPsych CDN unreachable: {e}")


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=str(CHROMIUM))
        yield b
        b.close()


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    doc = trial_lists(Design.load(EXPERIMENT_ASSETS_DIR / "demo_design.json"), seed=41, n_lists=3, n_catch=2)
    exp = tmp_path_factory.mktemp("exp")
    index = build_live_site(doc, exp)
    html = index.read_text()
    assert ensure_submit_bridge(html) == html  # the page posts to /submit itself
    index.write_text(ensure_consent_gate(html, load_consent_html()))
    (exp / "experiment" / "auto_psych_config.json").write_text(json.dumps(
        {"collection_session_id": "cs-test", "project_id": "rsa_reference", "prolific_redirect_url": REDIRECT}))
    return doc, exp / "experiment"


def _serve(page, cdn, site: Path, *, assign_status=200, submit_status=200):
    calls = {"assign": [], "submit": [], "blocked": []}

    def handle(route):
        req = route.request
        url = req.url
        if url in cdn:
            ctype, body = cdn[url]
            return route.fulfill(status=200, body=body, headers={"Content-Type": ctype})
        if url == f"{ORIGIN}/assign":
            calls["assign"].append(json.loads(req.post_data))
            if assign_status != 200:
                return route.fulfill(status=assign_status, body="no")
            return route.fulfill(status=200, body=json.dumps({"list_index": 2}),
                                 headers={"Content-Type": "application/json"})
        if url == f"{ORIGIN}/submit":
            calls["submit"].append(json.loads(req.post_data))
            return route.fulfill(status=submit_status, body="OK" if submit_status == 200 else "Write failed")
        if url == REDIRECT:
            return route.fulfill(status=200, body="<html><body>completed</body></html>",
                                 headers={"Content-Type": "text/html"})
        if url.startswith(f"{ORIGIN}/e1/"):
            path = site / url[len(f"{ORIGIN}/e1/"):].split("?")[0]
            if path.is_file():
                ctype = {".html": "text/html", ".json": "application/json", ".png": "image/png"}.get(path.suffix,
                                                                                                    "text/plain")
                return route.fulfill(status=200, body=path.read_bytes(), headers={"Content-Type": ctype})
        calls["blocked"].append(url)
        return route.abort()

    page.route("**/*", handle)
    return calls


def _participate(page, query: str):
    page.goto(f"{ORIGIN}/e1/index.html?{query}")
    page.wait_for_selector(f"#{CONSENT_GATE_MARKER}-agree")
    page.click(f"#{CONSENT_GATE_MARKER}-agree")
    for _ in range(200):
        if page.url == REDIRECT or page.locator(".rsa-error").count():
            return
        try:
            page.wait_for_function("document.querySelector('.rsa-ref, .rsa-next, .rsa-error') !== null"
                                   " || location.href.indexOf('/done') >= 0", timeout=15000)
            if page.locator(".rsa-ref").count():
                page.locator(".rsa-ref").first.click()
            elif page.locator(".rsa-next").count():
                page.locator(".rsa-next").first.click()
            page.wait_for_timeout(50)
        except Exception:  # the redirect replaced the page mid-step
            page.wait_for_timeout(200)
    raise AssertionError("the participant never finished")


def test_a_participant_consents_gets_the_assigned_list_submits_and_returns_to_prolific(browser, cdn, staged):
    doc, site = staged
    page = browser.new_context().new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    calls = _serve(page, cdn, site)
    _participate(page, "PROLIFIC_PID=P1&STUDY_ID=S1&SESSION_ID=X1&participant_id=P1")
    page.wait_for_url(REDIRECT, timeout=15000)
    assert not calls["blocked"] and not errors
    assert calls["assign"] == [{"collection_session_id": "cs-test", "n_lists": 3, "participant_key": "P1"}]
    (payload,) = calls["submit"]
    assert payload["collection_session_id"] == "cs-test" and payload["list_index"] == 2
    assert (payload["prolific_pid"], payload["prolific_study_id_from_url"], payload["prolific_session_id"]) == (
        "P1", "S1", "X1")
    assert payload["consented_at"]
    tasks = [r.get("task") for r in payload["trials"]]
    assert "consent" not in tasks  # the deployment's IRB gate replaces the page's own screen
    assert {r["list_index"] for r in payload["trials"]} == {2}
    assert {r["list_assignment"] for r in payload["trials"]} == {"server"}
    frame = convert(json.dumps(payload["trials"]), participant_id="p0", experiment="live_test")
    tests = [t for t in doc["lists"][2]["trials"] if t["phase"] == "test"]
    assert list(frame["condition"]) == [("catch" if t["is_catch"] else t["condition"]) for t in tests]


def test_without_a_participant_id_the_page_stops_rather_than_pick_a_list(browser, cdn, staged):
    _, site = staged
    page = browser.new_context().new_page()
    calls = _serve(page, cdn, site)
    page.goto(f"{ORIGIN}/e1/index.html")
    page.wait_for_selector(".rsa-error")
    assert "no participant id" in page.locator(".rsa-error").inner_text()
    assert calls["assign"] == [] and calls["submit"] == []


def test_a_failed_submit_shows_an_error_and_does_not_redirect(browser, cdn, staged):
    _, site = staged
    page = browser.new_context().new_page()
    calls = _serve(page, cdn, site, submit_status=500)
    _participate(page, "PROLIFIC_PID=P2&STUDY_ID=S1&SESSION_ID=X2")
    page.wait_for_selector(".rsa-error")
    assert "could not be submitted" in page.locator(".rsa-error").inner_text()
    assert page.url != REDIRECT and len(calls["submit"]) == 1


def test_the_live_page_carries_its_hooks_once():
    doc = trial_lists(Design.load(EXPERIMENT_ASSETS_DIR / "demo_design.json"), seed=1, n_lists=2, n_catch=0)
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        html = build_live_site(doc, Path(d)).read_text()
    assert html.count(LIVE_MARKER) >= 1 and html.index(LIVE_MARKER) < html.index("const TRIAL_LISTS")
    assert "const CONSENT_HTML = null;" in html
