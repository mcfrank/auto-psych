"""Fail-loud policy for the browser-steering and live-collection paths.

Policy (project rule): an *unexpected* failure raises; an *expected* outcome is
handled by an explicit policy the caller can see in the log. ``collect.py``
used to break both halves of that rule with paired ``except Exception: pass``
blocks and bare ``return []`` paths:

- steering could fail to click *and* fail to press a key, after which the drive
  loop simply spun to its 3-minute timeout with nothing written anywhere. This
  is the exact shape of a bug the project already hit (steering pressed ``f``/
  ``j`` while the trials rendered buttons, yielding degenerate all-left data);
- a missing ``prolific_study_id`` / ``results_api_url``, or a failed ``/results``
  fetch, returned ``[]`` — indistinguishable from "nobody responded yet".

These tests pin the loud behavior.
"""

from __future__ import annotations

import urllib.error

import pytest

from src.pipelines.outer_loop import collect


# ── Playwright doubles ──────────────────────────────────────────────────────


def _playwright_error(message: str) -> Exception:
    from playwright.sync_api import Error as PlaywrightError

    return PlaywrightError(message)


class _FakeLocator:
    """Stands in for a Playwright locator over ``button.jspsych-btn``."""

    def __init__(self, count: int, click_error: Exception | None = None):
        self._count = count
        self._click_error = click_error
        self.clicked: list[int] = []

    def count(self) -> int:
        return self._count

    def nth(self, index: int) -> "_FakeButton":
        return _FakeButton(self, index)

    @property
    def first(self) -> "_FakeButton":
        return _FakeButton(self, 0)


class _FakeButton:
    def __init__(self, locator: _FakeLocator, index: int):
        self._locator = locator
        self._index = index

    def click(self, timeout: int | None = None) -> None:
        if self._locator._click_error is not None:
            raise self._locator._click_error
        self._locator.clicked.append(self._index)


class _FakeKeyboard:
    def __init__(self, error: Exception | None = None):
        self._error = error
        self.pressed: list[str] = []

    def press(self, key: str) -> None:
        if self._error is not None:
            raise self._error
        self.pressed.append(key)


class _FakePage:
    def __init__(
        self,
        *,
        n_buttons: int = 0,
        click_error: Exception | None = None,
        key_error: Exception | None = None,
        evaluate_error: Exception | None = None,
        evaluate_result=None,
    ):
        self.locator_obj = _FakeLocator(n_buttons, click_error)
        self.keyboard = _FakeKeyboard(key_error)
        self._evaluate_error = evaluate_error
        self._evaluate_result = evaluate_result
        self.closed = False

    def locator(self, selector: str) -> _FakeLocator:
        return self.locator_obj

    def evaluate(self, script: str):
        if self._evaluate_error is not None:
            raise self._evaluate_error
        return self._evaluate_result

    def close(self) -> None:
        self.closed = True


# ── Steering: a choice that cannot be applied must not vanish ───────────────


def test_click_random_choice_raises_when_neither_modality_works():
    page = _FakePage(
        n_buttons=2,
        click_error=_playwright_error("element is not visible"),
        key_error=_playwright_error("target page closed"),
    )
    with pytest.raises(RuntimeError, match="could not advance"):
        collect._click_random_choice(page)


def test_click_random_choice_logs_the_keyboard_fallback(capsys):
    page = _FakePage(n_buttons=2, click_error=_playwright_error("intercepted"))
    collect._click_random_choice(page)
    assert page.keyboard.pressed  # the fallback actually happened
    err = capsys.readouterr().err
    assert "click" in err.lower() and "key" in err.lower()


def test_click_random_choice_on_a_keyboard_trial_stays_quiet(capsys):
    page = _FakePage(n_buttons=0)
    collect._click_random_choice(page)
    assert page.keyboard.pressed
    assert capsys.readouterr().err == ""  # no buttons is normal, not a fallback


def test_click_random_choice_does_not_let_a_working_keypress_hide_a_bug():
    # The historical failure exactly: the button modality is broken, the key
    # press "succeeds", and the run completes with every trial decided by a key
    # the trials never listened for. A non-Playwright error there is a bug in
    # this module and must not be papered over by the fallback.
    page = _FakePage(n_buttons=2, click_error=AttributeError("no attribute 'nth'"))
    with pytest.raises(AttributeError):
        collect._click_random_choice(page)
    assert not page.keyboard.pressed


def test_act_key_raises_when_neither_modality_works():
    page = _FakePage(
        n_buttons=2,
        click_error=_playwright_error("element is not visible"),
        key_error=_playwright_error("target page closed"),
    )
    with pytest.raises(RuntimeError, match="could not apply"):
        collect._act_key(page, "f")


def test_act_key_still_translates_left_to_the_first_button():
    page = _FakePage(n_buttons=3)
    collect._act_key(page, "f")
    assert page.locator_obj.clicked == [0]


# ── Screen reading: only page-level errors are tolerated ────────────────────


def test_get_screen_content_tolerates_a_page_level_error(capsys):
    page = _FakePage(evaluate_error=_playwright_error("Execution context destroyed"))
    assert collect._get_screen_content(page) == ""
    assert "screen" in capsys.readouterr().err.lower()


def test_get_screen_content_propagates_programming_errors():
    page = _FakePage(evaluate_error=TypeError("evaluate() got an unexpected kwarg"))
    with pytest.raises(TypeError):
        collect._get_screen_content(page)


# ── LLM steering: a missing key/dep is a config error, not "blind mode" ─────


def test_drive_with_llm_propagates_llm_configuration_failure(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("GOOGLE_API_KEY is not set")

    monkeypatch.setattr(collect, "get_llm", boom)
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        collect._drive_experiment_with_llm(
            _FakePage(), timeout_ms=10, project_id="p", run_id=1, logs_dir=tmp_path
        )


def test_drive_with_llm_reports_a_missing_steering_prompt(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(collect, "get_llm", lambda *a, **k: object())
    monkeypatch.setattr(collect, "load_prompt_for_run", lambda *a, **k: "   ")
    done, llm_used = collect._drive_experiment_with_llm(
        _FakePage(), timeout_ms=10, project_id="p", run_id=1, logs_dir=tmp_path
    )
    assert (done, llm_used) == (False, False)
    assert "4_collect_steering" in capsys.readouterr().err


# ── Live collection: never return [] for a config or fetch failure ─────────


def _live_state():
    return {"project_id": "subjective_randomness", "run_id": 1}


def test_collect_live_raises_when_prolific_study_id_is_missing(tmp_path):
    with pytest.raises(RuntimeError, match="prolific_study_id"):
        collect._collect_live(
            _live_state(),
            {"results_api_url": "https://example.invalid"},
            tmp_path,
            tmp_path,
        )


def test_collect_live_raises_when_results_url_is_missing(tmp_path):
    with pytest.raises(RuntimeError, match="results_api_url"):
        collect._collect_live(
            _live_state(), {"prolific_study_id": "study1"}, tmp_path, tmp_path
        )


def test_collect_live_raises_when_the_results_fetch_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "_poll_prolific_until_target", lambda *a, **k: 1)

    def boom(*args, **kwargs):
        raise urllib.error.URLError("gateway timeout")

    monkeypatch.setattr(collect.urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError, match="results fetch failed"):
        collect._collect_live(
            _live_state(),
            {
                "prolific_study_id": "study1",
                "results_api_url": "http://example.invalid",
            },
            tmp_path,
            tmp_path,
        )


def test_collect_from_firebase_raises_when_the_results_fetch_fails(
    tmp_path, monkeypatch
):
    def boom(*args, **kwargs):
        raise urllib.error.URLError("gateway timeout")

    monkeypatch.setattr(collect.urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError, match="results fetch failed"):
        collect._collect_from_firebase(
            _live_state(),
            {
                "collection_session_id": "sess1",
                "project_id": "subjective_randomness",
                "run_id": 1,
            },
            "http://example.invalid",
            0,
            tmp_path,
            tmp_path,
        )


# ── Malformed stimuli must not become blank options ────────────────────────


def test_llm_participant_rows_reject_a_stimulus_without_sequences():
    class _Model:
        name = "fake"

        def answer(self, system, user):  # pragma: no cover - never reached
            return "ANSWER: left"

    with pytest.raises(ValueError, match="sequence_a"):
        collect.generate_llm_participant_rows(
            [{"sequence_a": "HT"}],
            1,
            participant_model=_Model(),
            prompt_text="choose",
        )
