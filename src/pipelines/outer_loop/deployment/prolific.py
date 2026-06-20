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


# Prolific actions valid for a COMPLETED completion code. Auto-approve pays every
# worker who finishes (the Prolific norm — data quality is an *analysis* decision,
# handled by the monitor, not a payment one); manual review holds submissions for
# vetting. Other API actions (group add/remove, screen-out payment, …) need extra
# fields and don't apply to a plain completion, so we reject them loudly.
_COMPLETION_ACTIONS = {"AUTOMATICALLY_APPROVE", "MANUALLY_REVIEW"}
DEFAULT_COMPLETION_ACTION = "AUTOMATICALLY_APPROVE"


def completion_redirect_url(code: str) -> str:
    return "https://app.prolific.com/submissions/complete?cc=" + urllib.parse.quote(code)


# Data-quality eligibility defaults applied to every real-recruitment study so
# we collect from US-based, English-fluent participants with a strong approval
# history. The choice IDs are Prolific's stable identifiers from
# GET /api/v1/filters/ (NOT display order):
#   current-country-of-residence "1"  -> United States
#   fluent-languages             "19" -> English
DEFAULT_MIN_APPROVAL_RATE = 98
UNITED_STATES_RESIDENCE_CHOICE_ID = "1"
ENGLISH_FLUENT_LANGUAGE_CHOICE_ID = "19"


def verify_eligibility_choice_ids(filters: list[dict[str, Any]]) -> None:
    """Assert Prolific's choice IDs still mean what we hardcode in
    ``build_eligibility_filters``.

    The IDs are stable in practice, but a silent remap would make us recruit the
    wrong pool, so before a live run we check them against Prolific's current
    ``GET /filters/`` payload (passed in as ``filters``). Raises ``ValueError``
    on any drift or missing filter — recruiting against unverified IDs is exactly
    the silent fallback we want to avoid.
    """
    expected = {
        "current-country-of-residence": (UNITED_STATES_RESIDENCE_CHOICE_ID, "United States"),
        "fluent-languages": (ENGLISH_FLUENT_LANGUAGE_CHOICE_ID, "English"),
    }
    by_id = {f.get("filter_id"): f for f in filters}
    for filter_id, (choice_id, expected_label) in expected.items():
        spec = by_id.get(filter_id)
        if spec is None:
            raise ValueError(
                f"Prolific filter {filter_id!r} is missing from GET /filters/; "
                "cannot confirm eligibility choice IDs before recruiting."
            )
        actual_label = (spec.get("choices") or {}).get(choice_id)
        if actual_label != expected_label:
            raise ValueError(
                f"Prolific choice ID drift: {filter_id!r} choice {choice_id!r} is "
                f"{actual_label!r}, expected {expected_label!r}. The hardcoded "
                "eligibility IDs are stale — re-check GET /filters/ before recruiting."
            )


def build_eligibility_filters(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Prolific eligibility filters enforcing our data-quality baseline:
    US residence, English fluency, and a minimum approval rate.

    The approval-rate floor is overridable via the ``min_approval_rate`` config
    key (a percentage, default ``DEFAULT_MIN_APPROVAL_RATE``); residence and
    language use the documented defaults. An out-of-range approval rate raises
    rather than silently creating a study that recruits the wrong pool.
    """
    min_approval_rate = int(cfg.get("min_approval_rate", DEFAULT_MIN_APPROVAL_RATE))
    if not 0 <= min_approval_rate <= 100:
        raise ValueError(
            f"min_approval_rate must be a percentage between 0 and 100, got "
            f"{min_approval_rate}."
        )
    return [
        {
            "filter_id": "current-country-of-residence",
            "selected_values": [UNITED_STATES_RESIDENCE_CHOICE_ID],
        },
        {
            "filter_id": "fluent-languages",
            "selected_values": [ENGLISH_FLUENT_LANGUAGE_CHOICE_ID],
        },
        {
            "filter_id": "approval_rate",
            "selected_range": {"lower": min_approval_rate, "upper": 100},
        },
    ]


def compute_reward_cents(cfg: dict[str, Any]) -> int:
    """Reward (cents) for a study, derived from the configured hourly wage.

    When ``reward_per_hour`` (cents/hour) is set, the reward is computed from it
    and ``estimated_completion_time`` (minutes) so the effective wage stays fixed
    even if the study length changes — e.g. 1200 cents/hr over 5 minutes is 100
    cents. Falls back to an explicit ``reward`` (cents), then to 50.

    Refuses to return a non-positive reward: a misconfigured wage (zero/negative
    ``reward_per_hour`` or ``estimated_completion_time``, or a per-hour rate so low
    it rounds to 0 cents) would otherwise create a study that pays real
    participants nothing. Raising here surfaces the problem at dry-run time, before
    any study is created.
    """
    if cfg.get("reward_per_hour") is not None:
        reward_per_hour = float(cfg["reward_per_hour"])
        minutes = float(cfg.get("estimated_completion_time") or 5)
        if reward_per_hour <= 0 or minutes <= 0:
            raise ValueError(
                f"Prolific reward config is non-positive (reward_per_hour="
                f"{reward_per_hour} cents/hr, estimated_completion_time={minutes} "
                "min); refusing to create a study that underpays participants."
            )
        reward = round(reward_per_hour * minutes / 60.0)
    else:
        reward = int(cfg.get("reward") or 50)
    if reward <= 0:
        raise ValueError(
            f"Computed Prolific reward is {reward} cents; refusing to create a study "
            "that pays participants nothing. Fix reward_per_hour / reward / "
            "estimated_completion_time in the Prolific config."
        )
    return reward


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
    completion_action = str(cfg.get("completion_code_action") or DEFAULT_COMPLETION_ACTION)
    if completion_action not in _COMPLETION_ACTIONS:
        raise ValueError(
            f"completion_code_action must be one of {sorted(_COMPLETION_ACTIONS)}, "
            f"got {completion_action!r}."
        )
    payload: dict[str, Any] = {
        "name": cfg.get("name") or f"Auto-psych {manifest.experiment_id}",
        "internal_name": f"auto-psych {manifest.deployment_id}",
        "description": cfg.get("description") or "Psychology experiment (auto-psych pipeline).",
        "external_study_url": external_study_url(manifest.experiment_url),
        "prolific_id_option": "url_parameters",
        # Prolific's current API: an array of completion codes, each with a type
        # and automatic actions. The participant is redirected to `?cc=<code>`
        # (see `redirect` above), so this code must equal that one.
        "completion_codes": [
            {
                "code": completion_code,
                "code_type": "COMPLETED",
                "actions": [{"action": completion_action}],
            }
        ],
        "estimated_completion_time": int(cfg.get("estimated_completion_time") or 5),
        "total_available_places": int(cfg.get("total_available_places") or n_participants),
        "reward": compute_reward_cents(cfg),
        "device_compatibility": cfg.get("device_compatibility") or ["desktop"],
    }
    if mode == "test" and test_participant_id:
        # Test preview: pin recruitment to the single known test participant.
        # The data-quality eligibility filters are intentionally skipped here —
        # the allowlist already restricts to one account, which need not carry
        # the demographics the live filters require.
        payload["filters"] = [
            {
                "filter_id": "custom_allowlist",
                "selected_values": [test_participant_id],
            }
        ]
        payload["total_available_places"] = 1
    else:
        payload["filters"] = build_eligibility_filters(cfg)
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
    from src.runtime.prolific import create_study, get_filters

    plan = build_prolific_plan(
        project_id=project_id,
        manifest=manifest,
        n_participants=n_participants,
        mode=mode,
    )
    # Live studies recruit paid participants gated by hardcoded choice IDs, so
    # confirm those IDs still mean what we think before any study is created.
    if mode == "live":
        filters, err = get_filters()
        if err:
            raise RuntimeError(
                f"Could not fetch Prolific filters to verify eligibility choice IDs "
                f"before recruiting: {err}"
            )
        verify_eligibility_choice_ids(filters)
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
