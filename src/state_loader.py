"""
Load pipeline state from an existing run (or reference run) for single-agent runs
and debugging. Optionally build minimal state for an agent using fixtures.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from src.config import (
    REPO_ROOT,
    project_dir,
    problem_definition_path,
    run_dir,
    agent_dir,
)


def load_state_from_run(
    project_id: str,
    run_id: int,
    reference_run_id: Optional[int] = None,
    mode: str = "simulated_participants",
) -> Dict[str, Any]:
    """
    Build PipelineState by scanning the run directory (or reference_run_id's dir)
    for existing artifacts. Used to re-run a single agent or run downstream agents
    in isolation using outputs from a previous run.
    """
    r_id = reference_run_id if reference_run_id is not None else run_id
    rdir = run_dir(project_id, r_id)
    prob_path = problem_definition_path(project_id)

    # Registry path is per run (current run_id); pipeline and run_agent both write model_registry.yaml there
    current_rdir = run_dir(project_id, run_id)
    state: Dict[str, Any] = {
        "project_id": project_id,
        "run_id": run_id,
        "mode": mode,
        "problem_definition_path": str(prob_path) if prob_path.exists() else "",
        "registry_path": str(current_rdir / "model_registry.yaml"),
    }

    # 1_theory
    mf = rdir / "1_theory" / "models_manifest.yaml"
    if mf.exists():
        state["theorist_manifest_path"] = str(mf)
    ra = rdir / "1_theory" / "rationale.md"
    if ra.exists():
        state["theorist_rationale_path"] = str(ra)

    # 2_design
    st = rdir / "2_design" / "stimuli.json"
    if st.exists():
        state["stimuli_path"] = str(st)
    dr = rdir / "2_design" / "design_rationale.md"
    if dr.exists():
        state["design_rationale_path"] = str(dr)

    # 3_implement (experiment_path = directory; config.json in same dir)
    exp_dir = rdir / "3_implement"
    if exp_dir.exists():
        state["experiment_path"] = str(exp_dir)
    cfg = rdir / "3_implement" / "config.json"
    if cfg.exists():
        state["deployment_config_path"] = str(cfg)

    # 4_collect
    resp = rdir / "4_collect" / "responses.csv"
    if resp.exists():
        state["simulated_data_path"] = str(resp)

    # 5_analyze
    ss = rdir / "5_analyze" / "summary_stats.json"
    if ss.exists():
        state["summary_stats_path"] = str(ss)
    ag = rdir / "5_analyze" / "aggregate.csv"
    if ag.exists():
        state["aggregate_csv_path"] = str(ag)

    # 6_interpret
    rep = rdir / "6_interpret" / "report.md"
    if rep.exists():
        state["interpreter_report_path"] = str(rep)

    return state


def minimal_state_for_agent(
    agent_key: str,
    project_id: str,
    run_id: int,
    fixtures_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Return minimal state required to run the given agent. Uses fixtures_dir
    for any missing artifacts (e.g. when no prior run exists). If fixtures_dir
    is None, uses REPO_ROOT / "tests" / "fixtures".
    """
    if fixtures_dir is None:
        fixtures_dir = REPO_ROOT / "tests" / "fixtures"
    fixtures_dir = Path(fixtures_dir)
    prob_path = problem_definition_path(project_id)

    # Base state
    state: Dict[str, Any] = {
        "project_id": project_id,
        "run_id": run_id,
        "mode": "simulated_participants",
        "problem_definition_path": str(prob_path) if prob_path.exists() else str(fixtures_dir / "problem_definition.md"),
    }

    rdir = run_dir(project_id, run_id)

    def _path(*segments: str) -> str:
        p = fixtures_dir.joinpath(*segments)
        if p.exists():
            return str(p)
        alt = rdir.joinpath(*segments)
        if alt.exists():
            return str(alt)
        return ""

    if agent_key == "1_theory":
        if not state["problem_definition_path"] or not Path(state["problem_definition_path"]).exists():
            state["problem_definition_path"] = str(fixtures_dir / "problem_definition.md")
        state["registry_path"] = str(rdir / "model_registry.yaml")
        return state

    if agent_key == "2_design":
        state["theorist_manifest_path"] = _path("1_theory", "models_manifest.yaml") or str(fixtures_dir / "models_manifest.yaml")
        return state

    if agent_key == "3_implement":
        state["theorist_manifest_path"] = _path("1_theory", "models_manifest.yaml") or str(fixtures_dir / "models_manifest.yaml")
        state["stimuli_path"] = _path("2_design", "stimuli.json") or str(fixtures_dir / "stimuli.json")
        return state

    if agent_key == "4_collect":
        state["stimuli_path"] = _path("2_design", "stimuli.json") or str(fixtures_dir / "stimuli.json")
        state["theorist_manifest_path"] = _path("1_theory", "models_manifest.yaml") or str(fixtures_dir / "models_manifest.yaml")
        state["deployment_config_path"] = _path("3_implement", "config.json") or str(rdir / "3_implement" / "config.json")
        return state

    if agent_key == "5_analyze":
        state["deployment_config_path"] = _path("3_implement", "config.json") or str(rdir / "3_implement" / "config.json")
        state["theorist_manifest_path"] = _path("1_theory", "models_manifest.yaml") or str(fixtures_dir / "models_manifest.yaml")
        resp = rdir / "4_collect" / "responses.csv"
        if resp.exists():
            state["simulated_data_path"] = str(resp)
        return state

    if agent_key == "6_interpret":
        state["summary_stats_path"] = _path("5_analyze", "summary_stats.json") or str(rdir / "5_analyze" / "summary_stats.json")
        state["aggregate_csv_path"] = _path("5_analyze", "aggregate.csv") or str(rdir / "5_analyze" / "aggregate.csv")
        state["theorist_manifest_path"] = _path("1_theory", "models_manifest.yaml") or str(fixtures_dir / "models_manifest.yaml")
        return state

    return state
