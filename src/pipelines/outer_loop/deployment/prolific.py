"""Pipeline-level Prolific orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import urllib.parse

from .manifest import DeploymentManifest


@dataclass
class ProlificStudyPlan:
    payload: dict[str, Any]
    completion_code: str
    redirect_url: str
    study_id: str | None = None
    test_participant_id: str | None = None
    published: bool = False


def completion_redirect_url(code: str) -> str:
    return "https://app.prolific.com/submissions/complete?cc=" + urllib.parse.quote(code)


def compute_reward_cents(cfg: dict[str, Any]) -> int:
    """Reward (cents) for a study, derived from the configured hourly wage.

    When ``reward_per_hour`` (cents/hour) is set, the reward is computed from it
    and ``estimated_completion_time`` (minutes) so the effective wage stays fixed
    even if the study length changes — e.g. 1200 cents/hr over 5 minutes is 100
    cents. Falls back to an explicit ``reward`` (cents), then to 50.
    """
    if cfg.get("reward_per_hour") is not None:
        minutes = float(cfg.get("estimated_completion_time") or 5)
        return round(float(cfg["reward_per_hour"]) * minutes / 60.0)
    return int(cfg.get("reward") or 50)


def external_study_url(base_url: str) -> str:
    sep = "&" if "?" in base_url else "?"
    return (
        f"{base_url}{sep}"
        "participant_id={{%PROLIFIC_PID%}}"
        "&PROLIFIC_PID={{%PROLIFIC_PID%}}"
        "&STUDY_ID={{%STUDY_ID%}}"
        "&SESSION_ID={{%SESSION_ID%}}"
    )


def build_prolific_plan(
    *,
    project_id: str,
    manifest: DeploymentManifest,
    n_participants: int,
    mode: str,
    test_participant_id: str | None = None,
) -> ProlificStudyPlan:
    from src.runtime.prolific import load_prolific_config

    if not manifest.experiment_url:
        raise ValueError("Prolific study creation requires an experiment_url")

    cfg = load_prolific_config(project_id)
    completion_code = str(cfg.get("completion_code") or "AUTO_PSYCH_COMPLETE")
    redirect = str(cfg.get("prolific_redirect_url") or completion_redirect_url(completion_code))
    payload: dict[str, Any] = {
        "name": cfg.get("name") or f"Auto-psych {manifest.experiment_id}",
        "internal_name": f"auto-psych {manifest.deployment_id}",
        "description": cfg.get("description") or "Psychology experiment (auto-psych pipeline).",
        "external_study_url": external_study_url(manifest.experiment_url),
        "prolific_id_option": "url_parameters",
        "completion_code": completion_code,
        "completion_option": "code",
        "estimated_completion_time": int(cfg.get("estimated_completion_time") or 5),
        "total_available_places": int(cfg.get("total_available_places") or n_participants),
        "reward": compute_reward_cents(cfg),
        "device_compatibility": cfg.get("device_compatibility") or ["desktop"],
    }
    if mode == "test" and test_participant_id:
        payload["filters"] = [
            {
                "filter_id": "custom_allowlist",
                "selected_values": [test_participant_id],
            }
        ]
        payload["total_available_places"] = 1
    return ProlificStudyPlan(
        payload=payload,
        completion_code=completion_code,
        redirect_url=redirect,
        test_participant_id=test_participant_id,
    )


def create_draft_study(project_id: str, manifest: DeploymentManifest, n_participants: int, mode: str) -> ProlificStudyPlan:
    """Create a DRAFT Prolific study. It is never published here — the caller
    publishes only for live mode. Test mode creates the same draft (no test
    participant) so you can preview it in Prolific with a made-up PROLIFIC_PID.
    """
    from src.runtime.prolific import create_study

    plan = build_prolific_plan(
        project_id=project_id,
        manifest=manifest,
        n_participants=n_participants,
        mode=mode,
    )
    study_id, err = create_study(plan.payload)
    if err:
        raise RuntimeError(f"Failed to create Prolific study: {err}")
    plan.study_id = study_id
    return plan


def publish_study(plan: ProlificStudyPlan) -> ProlificStudyPlan:
    if not plan.study_id:
        raise ValueError("Cannot publish Prolific study without study_id")
    from src.runtime.prolific import publish_study as publish_study_api

    ok, err = publish_study_api(plan.study_id)
    if not ok:
        raise RuntimeError(f"Failed to publish Prolific study: {err}")
    plan.published = True
    return plan
