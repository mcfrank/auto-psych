"""Pipeline-level Prolific orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
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
    return "https://app.prolific.com/submissions/complete?cc=" + urllib.parse.quote(
        code
    )


# Data-quality eligibility defaults applied to every real-recruitment study so
# we collect from US-based, English-fluent participants with a strong approval
# history. The choice IDs are Prolific's stable identifiers from
# GET /api/v1/filters/ (NOT display order):
#   current-country-of-residence "1"  -> United States
#   fluent-languages             "19" -> English
DEFAULT_MIN_APPROVAL_RATE = 98
UNITED_STATES_RESIDENCE_CHOICE_ID = "1"
ENGLISH_FLUENT_LANGUAGE_CHOICE_ID = "19"

# Every study the pipeline creates is named ``auto-psych <deployment id>``
# internally; that prefix is how a later study finds the earlier ones.
STUDY_INTERNAL_NAME_PREFIX = "auto-psych "
# Prolific's "Exclude participants from other studies" filter: a select over
# study IDs, evaluated when the study is published, so it excludes everyone
# who had taken part in a listed study by then (approved, returned, timed out
# or awaiting review). The listed studies must be in the new study's Prolific
# project; the pipeline creates every study in the account's default one.
EARLIER_STUDIES_BLOCKLIST_FILTER = "previous_studies_blocklist"


def verify_eligibility_choice_ids(filters: list[dict[str, Any]]) -> None:
    """Assert Prolific's choice IDs still mean what we hardcode in
    ``build_eligibility_filters``, and that the filter excluding earlier
    studies' participants (``EARLIER_STUDIES_BLOCKLIST_FILTER``) exists.

    The IDs are stable in practice, but a silent remap would make us recruit the
    wrong pool, so before a live run we check them against Prolific's current
    ``GET /filters/`` payload (passed in as ``filters``). Raises ``ValueError``
    on any drift or missing filter — recruiting against unverified IDs is exactly
    the silent fallback we want to avoid.
    """
    expected = {
        "current-country-of-residence": (
            UNITED_STATES_RESIDENCE_CHOICE_ID,
            "United States",
        ),
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
    blocklist = by_id.get(EARLIER_STUDIES_BLOCKLIST_FILTER)
    if blocklist is None or blocklist.get("data_type", "StudyID") != "StudyID":
        raise ValueError(
            f"Prolific filter {EARLIER_STUDIES_BLOCKLIST_FILTER!r} (exclude the "
            "participants of earlier studies, by study ID) is missing from GET "
            f"/filters/ or no longer takes study IDs ({blocklist!r}); a live study "
            "would recruit the earlier studies' participants again."
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


def load_recruitment_config(project_id: str, n_participants: int) -> dict[str, Any]:
    """The project's Prolific settings for a study that recruits ``n_participants``.

    ``--n-participants`` is the one participant count: it sets the design's N,
    the collection poll's target and the study's places. The rendered
    prolific_config.yaml carries no count of its own; one that does (an older
    render) must agree, and a missing file raises rather than creating a
    study from the loader's defaults.
    """
    from src.runtime.prolific import load_prolific_config, prolific_config_path

    if n_participants < 1:
        raise ValueError(f"--n-participants must be >= 1, got {n_participants}.")
    path = prolific_config_path(project_id)
    if not path.exists():
        raise FileNotFoundError(
            f"No Prolific study settings at {path}. Render them from your launcher "
            "config: python scripts/outer_loop_live/_pilot_config.py <config.yaml> "
            "--render-only (run_pilot.sh and start_full_run.sh do this for you)."
        )
    cfg = load_prolific_config(project_id)
    places = cfg.get("total_available_places")
    if places is not None and int(places) != n_participants:
        raise ValueError(
            f"{path} sets total_available_places: {places}, but this run was started "
            f"with --n-participants {n_participants}. The study would recruit and pay "
            f"{places} people while the design and the collection target use "
            f"{n_participants}. --n-participants is the participant count: delete "
            "total_available_places from that file (re-render it with "
            "_pilot_config.py --render-only) or pass the number you mean to recruit."
        )
    return cfg


def excludes_earlier_participants(cfg: dict[str, Any]) -> bool:
    """The ``exclude_earlier_participants`` setting (default true): a live
    study excludes everyone who took part in an earlier study of the
    pipeline. Set it to false only to recruit the same people on purpose."""
    value = cfg.get("exclude_earlier_participants", True)
    if not isinstance(value, bool):
        raise ValueError(
            f"exclude_earlier_participants must be true or false, got {value!r}."
        )
    return value


def earlier_pipeline_study_ids() -> list[str]:
    """IDs of the account's earlier pipeline studies that people could take
    part in: named ``STUDY_INTERNAL_NAME_PREFIX``, and published (a draft,
    ``UNPUBLISHED``, never had participants). Raises when Prolific cannot
    list them, so no study is created without the exclusion."""
    from src.runtime.prolific import list_studies

    studies, err = list_studies()
    if err:
        raise RuntimeError(
            "Could not list the account's Prolific studies to exclude earlier "
            f"participants; no study was created: {err}"
        )
    return [
        str(study["id"])
        for study in studies
        if str(study.get("internal_name") or "").startswith(STUDY_INTERNAL_NAME_PREFIX)
        and study.get("status") != "UNPUBLISHED"
    ]


def build_prolific_plan(
    *,
    project_id: str,
    manifest: DeploymentManifest,
    n_participants: int,
    mode: str,
    test_participant_id: str | None = None,
    excluded_study_ids: Sequence[str] = (),
) -> ProlificStudyPlan:
    if not manifest.experiment_url:
        raise ValueError("Prolific study creation requires an experiment_url")

    cfg = load_recruitment_config(project_id, n_participants)
    completion_code = str(cfg.get("completion_code") or "AUTO_PSYCH_COMPLETE")
    redirect = str(
        cfg.get("prolific_redirect_url") or completion_redirect_url(completion_code)
    )
    completion_action = str(
        cfg.get("completion_code_action") or DEFAULT_COMPLETION_ACTION
    )
    if completion_action not in _COMPLETION_ACTIONS:
        raise ValueError(
            f"completion_code_action must be one of {sorted(_COMPLETION_ACTIONS)}, "
            f"got {completion_action!r}."
        )
    payload: dict[str, Any] = {
        "name": cfg.get("name") or f"Auto-psych {manifest.experiment_id}",
        "internal_name": f"{STUDY_INTERNAL_NAME_PREFIX}{manifest.deployment_id}",
        "description": cfg.get("description")
        or "Psychology experiment (auto-psych pipeline).",
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
        "total_available_places": n_participants,
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
        if excluded_study_ids:
            payload["filters"].append(
                {
                    "filter_id": EARLIER_STUDIES_BLOCKLIST_FILTER,
                    "selected_values": list(excluded_study_ids),
                }
            )
    return ProlificStudyPlan(
        payload=payload,
        completion_code=completion_code,
        redirect_url=redirect,
        test_participant_id=test_participant_id,
    )


def verify_live_eligibility() -> None:
    """Check with Prolific that the hardcoded eligibility choice IDs still mean
    what the code assumes (US residence, English fluency). Read-only; raises
    on a failed request or on drift. A live deploy runs it before deploying
    anything, and again just before creating the study."""
    from src.runtime.prolific import get_filters

    filters, err = get_filters()
    if err:
        raise RuntimeError(
            f"Could not fetch Prolific filters to verify eligibility choice IDs "
            f"before recruiting: {err}"
        )
    verify_eligibility_choice_ids(filters)


def create_draft_study(
    project_id: str, manifest: DeploymentManifest, n_participants: int, mode: str
) -> ProlificStudyPlan:
    """Create a DRAFT Prolific study. It is never published here — the caller
    publishes only for live mode. Test mode creates the same draft (no test
    participant) so you can preview it in Prolific with a made-up PROLIFIC_PID.

    A live study excludes the participants of every earlier pipeline study in
    the account (``earlier_pipeline_study_ids``), unless the settings say
    ``exclude_earlier_participants: false``. The IDs are read now, just before
    the study is created and published, so they include the studies of
    parallel runs created so far.
    """
    from src.runtime.prolific import create_study

    exclude = excludes_earlier_participants(
        load_recruitment_config(project_id, n_participants)
    )
    excluded: list[str] = []
    # Live studies recruit paid participants gated by hardcoded choice IDs, so
    # confirm those IDs still mean what we think before any study is created.
    if mode == "live":
        verify_live_eligibility()
        if exclude:
            excluded = earlier_pipeline_study_ids()
    plan = build_prolific_plan(
        project_id=project_id,
        manifest=manifest,
        n_participants=n_participants,
        mode=mode,
        excluded_study_ids=excluded,
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
