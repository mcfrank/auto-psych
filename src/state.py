"""LangGraph state schema for the auto-psych pipeline."""

from pathlib import Path
from typing import TypedDict


class PipelineState(TypedDict, total=False):
    """State passed between nodes. Paths are relative to repo root or absolute."""

    project_id: str
    run_id: int
    mode: str  # "simulated_participants" | "live" | "test_prolific"

    # Paths to latest artifacts from each stage
    problem_definition_path: str
    theorist_manifest_path: str
    theorist_rationale_path: str
    stimuli_path: str
    design_rationale_path: str
    experiment_path: str
    deployment_config_path: str
    summary_stats_path: str
    aggregate_csv_path: str
    interpreter_report_path: str

    # Set by collect step for data_analyst to read
    simulated_data_path: str

    # Pipeline defaults (overridable via CLI)
    simulated_n_participants: int
    max_validation_retries: int

    # Validation loop: feedback for retry, result of last validation, retry count per agent
    validation_feedback: str
    validation_ok: bool
    validation_retry_count: int
    last_validated_agent: str  # agent_key of last validator run; used to reset retry count when switching agents

    # Multi-run: current run's model registry; merged data for interpreter
    registry_path: str
    merged_aggregate_path: str
    merged_summary_path: str
