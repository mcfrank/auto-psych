#!/usr/bin/env python3
"""End-to-end smoke test for the Firebase + Prolific deployment setup.

This drives the *real* deployment and collection code paths with simulated
participants, so it proves the pieces a Prolific run depends on actually work
together:

1. Render the real participant template (``templates/jspsych_experiment.html``)
   over a few subjective-randomness H/T stimuli into a smoke experiment dir.
2. Deploy it to Firebase Hosting + Cloud Functions (the same ``run_deployment``
   the pipeline uses). Hosting serves ``index.html`` + ``auto_psych_config.json``;
   the ``/submit`` and ``/results`` functions front Firestore.
3. Run simulated participants with **Playwright** (headless Chromium) driving the
   live hosted experiment, steered by **Gemini** (one ``ACTION:`` per screen).
   Each participant's ``onFinish`` POSTs trials to ``/submit`` → Firestore.
4. Verify the data round-trips: ``/results`` returns CSV rows for the
   participants we ran.

Prolific recruitment itself is intentionally NOT exercised (it costs money and
needs human approval); the simulated participants stand in for Prolific workers
hitting the same hosted URL. Pass ``--prolific-mode test`` to also create a
Prolific test study (needs PROLIFIC_API_TOKEN + a test participant email).

Prerequisites for the live (``--deploy-target firebase``) path:
  - ``GOOGLE_API_KEY`` set (env or repo-root ``.secrets``) — Gemini steering.
  - Firebase CLI auth: ``firebase login`` (or ``npx firebase-tools login``).
  - Playwright Chromium installed: ``uv run playwright install chromium``.

Because the live path deploys to production and writes to production Firestore,
it refuses to run without ``--confirm-production``.

Usage:

    # Validate everything except the live deploy (no credentials needed):
    uv run python scripts/smoke_firebase_deploy.py --deploy-target dry-run

    # Full live smoke (deploys + writes prod Firestore):
    export GOOGLE_API_KEY=...
    uv run python scripts/smoke_firebase_deploy.py \
        --confirm-production --n-participants 2 --n-stimuli 4

Exit codes: 0 = success; 2 = preflight failed; 3 = deploy failed;
4 = collection produced no rows.
"""

from __future__ import annotations

import csv
import json
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import tyro
from pyprojroot import here

REPO_ROOT = here()
sys.path.insert(0, str(REPO_ROOT))

# `_collect_from_firebase` is the production browser-collection routine (headless
# Chromium + Gemini steering → /submit → Firestore, then fetch /results). We call
# it directly so the smoke test always *drives* simulated participants, even when
# a Prolific study exists (which would otherwise route the pipeline to polling).
from src.pipelines.outer_loop.collect import _collect_from_firebase  # noqa: E402
from src.pipelines.outer_loop.deployment.local import run_deployment  # noqa: E402
from src.pipelines.outer_loop.deployment.manifest import load_manifest  # noqa: E402
from src.pipelines.outer_loop.deployment.smoke import (  # noqa: E402
    render_template_experiment,
)
from src.pipelines.outer_loop.llm import resolve_google_api_key  # noqa: E402

DEFAULT_FIREBASE_PROJECT = "auto-psych-2c5da"


@dataclass
class Args:
    """Smoke-test the Firebase deploy + simulated-participant collection."""

    deploy_target: Literal["firebase", "dry-run"] = "firebase"
    """`firebase` = real live deploy + collection; `dry-run` = stage artifacts only."""
    confirm_production: bool = False
    """Required for `--deploy-target firebase`: I understand this writes to production."""
    n_participants: int = 2
    """How many simulated participants to run through the experiment."""
    n_stimuli: int = 4
    """How many H/T sequence pairs to show each participant."""
    project_id: str = "subjective_randomness"
    """Project id used for bookkeeping and the steering/participant prompts."""
    run_id: int = 9999
    """Run id for this smoke deploy; kept distinct from real runs (e.g. 1)."""
    firebase_project: str = DEFAULT_FIREBASE_PROJECT
    """Firebase project to deploy to."""
    firebase_region: str = "us-central1"
    """Cloud Functions region."""
    collection_owner: str = "smoke"
    """Owner id recorded in the deployment manifest / Firestore docs."""
    prolific_mode: Literal["none", "test"] = "none"
    """`test` also creates a Prolific test study (needs PROLIFIC_API_TOKEN)."""
    exp_dir: Optional[Path] = None
    """Override the experiment dir (default: data/outer_loop/<project>/smoke_experiment)."""


def _fail(message: str, code: int) -> None:
    print(f"\nFAIL: {message}", file=sys.stderr, flush=True)
    sys.exit(code)


def _preflight(args: Args) -> None:
    """Check the production gate, plus Gemini + Playwright when running live.

    A dry-run only stages files, so it needs neither a Gemini key nor a browser
    — those checks are gated on the live (`firebase`) path.
    """
    print("== Preflight ==", flush=True)

    if args.deploy_target != "firebase":
        print("  dry-run: skipping Gemini / Playwright checks", flush=True)
        return

    if not args.confirm_production:
        _fail(
            "deploy-target 'firebase' deploys to the live project "
            f"'{args.firebase_project}' and writes to PRODUCTION Firestore.\n"
            "  Re-run with --confirm-production once you're ready, or validate\n"
            "  the artifacts first with --deploy-target dry-run.",
            2,
        )

    # Gemini is required for steering (raises a loud, actionable error if absent).
    resolve_google_api_key(require=True)
    print("  Gemini API key: found", flush=True)

    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
    except ImportError:
        _fail(
            "Playwright is not importable. Install it with:\n"
            "    uv sync && uv run playwright install chromium",
            2,
        )
    print("  Playwright: importable", flush=True)


def _resolve_exp_dir(args: Args) -> Path:
    if args.exp_dir is not None:
        return args.exp_dir
    return REPO_ROOT / "data" / "outer_loop" / args.project_id / "smoke_experiment"


def _deploy(args: Args, exp_dir: Path):
    """Render the template, deploy, and return the loaded deployment manifest."""
    print("\n== Render experiment ==", flush=True)
    experiment_dir = render_template_experiment(exp_dir, n_stimuli=args.n_stimuli)
    print(f"  Wrote {experiment_dir / 'index.html'}", flush=True)

    print(f"\n== Deploy ({args.deploy_target}) ==", flush=True)
    firebase_project = (
        args.firebase_project if args.deploy_target == "firebase" else None
    )
    try:
        manifest_path = run_deployment(
            exp_dir=exp_dir,
            project_id=args.project_id,
            run_id=args.run_id,
            deploy_target=args.deploy_target,
            prolific_mode=args.prolific_mode,
            agent_backend="smoke",
            collection_owner=args.collection_owner,
            firebase_project=firebase_project,
            firebase_region=args.firebase_region,
            n_participants=args.n_participants,
            repo_root=REPO_ROOT,
        )
    except Exception as exc:  # surface deploy/auth failures loudly
        _fail(
            f"deployment failed: {type(exc).__name__}: {exc}\n"
            "  If this is a Firebase auth error, run `firebase login` (or set a\n"
            "  FIREBASE_TOKEN) and confirm `npx firebase-tools projects:list` works.",
            3,
        )

    manifest = load_manifest(manifest_path)
    print(f"  Manifest: {manifest_path}", flush=True)
    print(f"  collection_session_id: {manifest.collection_session_id}", flush=True)
    print(f"  experiment_url: {manifest.experiment_url}", flush=True)
    return manifest


def _fetch_results_csv(results_api_url: str, collection_session_id: str) -> str:
    # /results is token-guarded; _results_request fails loudly if the shared
    # secret (AUTO_PSYCH_RESULTS_TOKEN) is missing from the environment.
    from src.pipelines.outer_loop.collect import _results_request

    query = urllib.parse.urlencode({"collection_session_id": collection_session_id})
    url = f"{results_api_url.rstrip('/')}/results?{query}"
    with urllib.request.urlopen(_results_request(url), timeout=60) as response:
        return response.read().decode("utf-8")


def _collect_and_verify(args: Args, exp_dir: Path, manifest) -> None:
    print("\n== Collect (Playwright + Gemini, live) ==", flush=True)
    print(
        f"  Running {args.n_participants} simulated participant(s) against "
        f"{manifest.experiment_url}",
        flush=True,
    )

    config = json.loads(
        (exp_dir / "experiment" / "config.json").read_text(encoding="utf-8")
    )
    data_dir = exp_dir / "data"
    logs_dir = data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "project_id": args.project_id,
        "run_id": args.run_id,
        "mode": "simulated_participants",
    }
    rows = _collect_from_firebase(
        state,
        config,
        str(config["results_api_url"]),
        args.n_participants,
        data_dir,
        logs_dir,
    )

    csv_path = data_dir / "responses.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    participants = {
        row.get("participant_id_str") or row.get("participant_id") for row in rows
    }
    print("\n== Verify ==", flush=True)
    print(f"  {csv_path}: {len(rows)} rows from {len(participants)} participant(s)", flush=True)

    if not rows:
        _fail(
            "no response rows came back from Firestore /results. The deploy may\n"
            "  have succeeded but participants did not submit (check Gemini key in\n"
            "  worker subprocesses, the hosted URL, and "
            f"{exp_dir / 'data' / 'logs'}).",
            4,
        )

    # Independent confirmation straight from the live /results endpoint.
    try:
        raw = _fetch_results_csv(
            manifest.results_api_url or manifest.experiment_url,
            manifest.collection_session_id,
        )
        n_result_rows = max(0, len(raw.strip().splitlines()) - 1)
        print(f"  /results endpoint returned {n_result_rows} data row(s) directly", flush=True)
    except Exception as exc:
        print(f"  (could not re-fetch /results directly: {exc})", flush=True)

    print(
        "\nThis smoke run wrote to production Firestore at:\n"
        f"  collection_sessions/{manifest.collection_session_id}/responses/*\n"
        f"  participant ids: {exp_dir / 'data' / 'logs' / 'participant_ids.txt'}\n"
        "  Delete this session in the Firebase console if you don't want the data.",
        flush=True,
    )
    print(f"\nPASS: {len(rows)} rows round-tripped through the deployed setup.", flush=True)


def main(args: Args) -> None:
    exp_dir = _resolve_exp_dir(args)
    print(f"Smoke experiment dir: {exp_dir}", flush=True)

    _preflight(args)
    manifest = _deploy(args, exp_dir)

    if args.deploy_target == "dry-run":
        public_dir = exp_dir / "deployment" / "public"
        print(
            "\nDry-run complete. Staged artifacts (no live deploy, no collection):\n"
            f"  {public_dir / 'index.html'}\n"
            f"  {public_dir / 'auto_psych_config.json'}\n"
            "Re-run with --deploy-target firebase --confirm-production for the live test.",
            flush=True,
        )
        print("\nPASS: dry-run staged a deployable experiment.", flush=True)
        return

    _collect_and_verify(args, exp_dir, manifest)


if __name__ == "__main__":
    main(tyro.cli(Args))
