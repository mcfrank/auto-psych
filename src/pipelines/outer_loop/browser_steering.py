"""Playwright-based browser steering for the collection stage."""

from __future__ import annotations

import random
import re
import sys
import time
from pathlib import Path

from src.pipelines.outer_loop.llm import get_llm, invoke_llm, load_prompt_for_run

_DRIVE_INTERVAL_MS = 1200
_DRIVE_TIMEOUT_MS = 180_000
_LLM_CONTEXT_MAX_SCREENS = 20
_FIXATION_WAIT_SEC = 0.25


def _playwright_errors() -> tuple[type[BaseException], ...]:
    """Exception types Playwright raises for *page-level* trouble.

    ``playwright.sync_api.Error`` (whose ``TimeoutError`` is a subclass) covers
    the outcomes the steering loop legitimately has to survive: the page
    navigated mid-call so the execution context was destroyed, an element went
    stale, the page/browser was closed. Everything else — a TypeError from a
    wrong call signature, an AttributeError from a renamed helper — is a bug in
    this module and must propagate rather than be read as "the screen was
    empty".
    """
    from playwright.sync_api import Error as PlaywrightError

    return (PlaywrightError,)


def _experiment_data_present(page) -> bool:
    """Has jsPsych published ``window.__experimentData`` yet?

    Polled every drive-loop tick. A Playwright page error here means the page
    is mid-navigation, in which case the data provably is not readable yet, so
    ``False`` is the correct answer (and logging it once a tick would bury the
    real messages). Any other exception propagates.
    """
    try:
        return bool(page.evaluate("typeof window.__experimentData !== 'undefined'"))
    except _playwright_errors():
        return False


def _get_screen_content(page) -> str:
    """Return the visible experiment text, or ``""`` when it cannot be read.

    Only a Playwright page error is tolerated (see :func:`_playwright_errors`),
    and it is logged: a screen that silently reads as empty would send the
    steering LLM a blank prompt and, through it, an arbitrary choice.
    """
    try:
        return (
            page.evaluate(
                """() => {
              const sel = document.querySelector('#jspsych-content')
                || document.querySelector('.jspsych-content-wrapper')
                || document.querySelector('.jspsych-display-element')
                || document.body;
              return sel ? (sel.innerText || sel.textContent || '').trim() : '';
            }"""
            )
            or ""
        )
    except _playwright_errors() as exc:
        print(
            f"  [steering] could not read the screen ({type(exc).__name__}: {exc}); "
            "treating it as empty",
            file=sys.stderr,
            flush=True,
        )
        return ""


def _parse_steering_action(text: str) -> tuple[str, str] | None:
    if not text or not isinstance(text, str):
        return None
    text = text.strip()
    match = re.search(r"ACTION:\s*click\s+(.+)", text, re.IGNORECASE | re.DOTALL)
    if match:
        label = match.group(1).strip().split("\n")[0].strip()
        if label:
            return ("click", label)
    match = re.search(
        r"ACTION:\s*key\s+(f|j|ArrowLeft|ArrowRight)", text, re.IGNORECASE
    )
    if not match:
        return None
    key = match.group(1)
    lowered = key.lower()
    if lowered == "f":
        return ("key", "f")
    if lowered == "j":
        return ("key", "j")
    if lowered == "arrowleft":
        return ("key", "ArrowLeft")
    if lowered == "arrowright":
        return ("key", "ArrowRight")
    return ("key", key)


# Directional keys the keyboard template listens for; used to translate an LLM's
# left/right decision into the correct button on a button-based trial.
_LEFT_KEYS = {"f", "arrowleft"}
_RIGHT_KEYS = {"j", "arrowright"}


def _press_key_or_raise(page, key: str, what: str, click_error: Exception | None) -> None:
    """Press ``key``; raise if that fails too, naming both failed modalities.

    The keyboard is the last resort after the button modality was tried. If it
    also fails the trial simply did not advance — and the historical behavior
    (swallow both errors) left the drive loop spinning to its 3-minute timeout
    with nothing logged, which is how a modality mismatch produced a whole run
    of degenerate one-sided data before anyone noticed.
    """
    try:
        page.keyboard.press(key)
    except Exception as exc:
        raise RuntimeError(
            f"{what}: neither a button click nor a {key!r} key press worked "
            f"(click: {click_error!r}; key: {type(exc).__name__}: {exc}). The "
            "trial did not advance — check that the steering modality matches "
            "the modality the experiment actually renders."
        ) from exc


def _click_random_choice(page) -> None:
    """Advance a trial by an UNBIASED random choice.

    When buttons are present, click a uniformly-random one (so a declined or
    unparseable LLM action never silently biases toward the first option);
    otherwise press a random directional key. A failed click falls back to the
    keyboard — but says so, and raises if the keyboard fails as well.
    """
    click_error: Exception | None = None
    try:
        buttons = page.locator("button.jspsych-btn")
        count = buttons.count()
        if count > 0:
            buttons.nth(random.randrange(count)).click(timeout=1000)
            return
    except _playwright_errors() as exc:
        # Only a Playwright page error earns the keyboard fallback. A bug in
        # this module must NOT be papered over by a key press that "works" —
        # that is how f/j got pressed at button trials for a whole run.
        click_error = exc
        print(
            f"  [steering] random button click failed ({type(exc).__name__}: {exc}); "
            "falling back to a random key press",
            file=sys.stderr,
            flush=True,
        )
    _press_key_or_raise(
        page, random.choice(["f", "j"]), "could not advance the trial", click_error
    )


def _act_key(page, key: str) -> None:
    """Apply a directional key choice, modality-aware.

    On a multi-button trial, translate left keys (f/ArrowLeft) to the first
    button and right keys (j/ArrowRight) to the last, so the participant's
    left/right decision lands on the right option whether the experiment uses
    keyboard or button responses. A single-button screen (consent/instructions)
    is advanced by clicking it. Falls back to an actual key press — loudly, and
    raising if that fails too (see :func:`_press_key_or_raise`).
    """
    lowered = key.lower()
    click_error: Exception | None = None
    try:
        buttons = page.locator("button.jspsych-btn")
        count = buttons.count()
        if count == 1:
            buttons.first.click(timeout=1000)
            return
        if count >= 2:
            if lowered in _LEFT_KEYS:
                idx = 0
            elif lowered in _RIGHT_KEYS:
                idx = count - 1
            else:
                idx = random.randrange(count)
            buttons.nth(idx).click(timeout=1000)
            return
    except _playwright_errors() as exc:  # see _click_random_choice
        click_error = exc
        print(
            f"  [steering] button click for key {key!r} failed "
            f"({type(exc).__name__}: {exc}); falling back to a key press",
            file=sys.stderr,
            flush=True,
        )
    _press_key_or_raise(page, key, f"could not apply the {key!r} choice", click_error)


def _drive_experiment_with_llm(
    page,
    timeout_ms: int,
    project_id: str,
    run_id: int,
    logs_dir: Path,
    state: dict | None = None,
) -> tuple[bool, bool]:
    """Steer one participant through the experiment with the LLM.

    Returns ``(finished, llm_used)``. ``llm_used=False`` now means exactly one
    thing — the project ships no ``4_collect_steering`` prompt, so there is
    nothing to steer with and the caller should fall back to blind clicking.
    An LLM that cannot be constructed (missing API key, missing dependency) is
    a *configuration* error and raises: silently demoting every simulated
    participant to blind random clicking produces data that looks collected but
    carries no signal.
    """
    llm = get_llm()

    steering_prompt = load_prompt_for_run(
        project_id, run_id, "4_collect_steering", state
    )
    if not steering_prompt.strip():
        print(
            f"  [steering] no 4_collect_steering prompt for project {project_id!r} "
            f"run {run_id}; falling back to blind clicking",
            file=sys.stderr,
            flush=True,
        )
        return (False, False)

    deadline = time.monotonic() + (timeout_ms / 1000.0)
    context_parts: list[str] = []
    while time.monotonic() < deadline:
        if _experiment_data_present(page):
            return (True, True)

        screen_text = _get_screen_content(page) or "(loading or empty screen)"
        if screen_text.strip() in ("+", ""):
            time.sleep(_FIXATION_WAIT_SEC)
            continue

        context_parts.extend(
            [
                "=== CURRENT SCREEN ===",
                screen_text,
                "",
                "Reply with exactly one line: ACTION: click <button label> or ACTION: key f|j|ArrowLeft|ArrowRight",
            ]
        )
        user_msg = "\n".join(context_parts)
        try:
            response = invoke_llm(
                system=steering_prompt,
                user=user_msg,
                llm=llm,
                source="browser_steering",
            )
        except Exception as exc:
            print(f"  [LLM steering error] {exc}", file=sys.stderr, flush=True)
            (logs_dir / "llm_steering_error.txt").write_text(str(exc), encoding="utf-8")
            return (False, True)

        action = _parse_steering_action(response)
        if action is None:
            # Unparseable reply: advance without biasing toward either side.
            print(
                f"  [steering] unparseable reply {response.strip()[:80]!r}; "
                "advancing with a random choice",
                file=sys.stderr,
                flush=True,
            )
            _click_random_choice(page)
        elif action[0] == "click":
            try:
                page.get_by_role("button", name=action[1]).click(timeout=2000)
            except _playwright_errors() as exc:
                # The LLM named a button this screen does not have. Expected
                # (it is generating free text), but never silent: an unnoticed
                # stream of these means every trial was decided by the RNG.
                print(
                    f"  [steering] no clickable button named {action[1]!r} "
                    f"({type(exc).__name__}); advancing with a random choice",
                    file=sys.stderr,
                    flush=True,
                )
                _click_random_choice(page)
        else:
            # Directional key: lands on the correct button for button trials.
            _act_key(page, action[1])

        context_parts.append(f"Your action: {response.strip()[:150]}")
        if len(context_parts) > _LLM_CONTEXT_MAX_SCREENS * 4:
            context_parts = context_parts[-(_LLM_CONTEXT_MAX_SCREENS * 4) :]
        time.sleep(0.1)

    return (_experiment_data_present(page), True)


def _drive_experiment_to_finish(page, timeout_ms: int = _DRIVE_TIMEOUT_MS) -> bool:
    """Blind fallback when LLM steering is unavailable: click through to the end.

    Picks an UNBIASED random option each step rather than always the first
    button, so a blind run does not systematically favor one side (which would
    silently produce degenerate, one-sided data).
    """
    deadline = time.monotonic() + (timeout_ms / 1000.0)
    step_sec = _DRIVE_INTERVAL_MS / 1000.0
    while time.monotonic() < deadline:
        if _experiment_data_present(page):
            return True
        _click_random_choice(page)
        time.sleep(step_sec)
    return _experiment_data_present(page)
