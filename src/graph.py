"""LangGraph workflow: 6-agent pipeline (theory → design → implement → collect → analyze → interpret)."""

from typing import Literal

from langgraph.graph import StateGraph, END

from src.config import DEFAULT_MAX_VALIDATION_RETRIES
from src.state import PipelineState
from src.agents.theorist import run_theorist
from src.agents.experiment_designer import run_experiment_designer
from src.agents.experiment_implementer import run_experiment_implementer
from src.agents.collect import run_collect
from src.agents.data_analyst import run_data_analyst
from src.agents.interpreter import run_interpreter
from src.validation.validators import run_validation, NODE_TO_AGENT_KEY

def _max_retries(state: PipelineState) -> int:
    """Max validation retries per agent (from state or default)."""
    return state.get("max_validation_retries", DEFAULT_MAX_VALIDATION_RETRIES)


def _validator_node(agent_key: str):
    """Return a node function that runs validation for the given agent_key."""
    def _run(state: PipelineState) -> PipelineState:
        return run_validation(state, agent_key)
    return _run


def _after_validate_theory(state: PipelineState) -> Literal["theory", "design"]:
    if not state.get("validation_ok", True) and state.get("validation_retry_count", 0) < _max_retries(state):
        return "theory"
    return "design"


def _after_validate_design(state: PipelineState) -> Literal["design", "implement"]:
    if not state.get("validation_ok", True) and state.get("validation_retry_count", 0) < _max_retries(state):
        return "design"
    return "implement"


def _after_validate_implement(state: PipelineState) -> Literal["implement", "collect", "analyze"]:
    if not state.get("validation_ok", True) and state.get("validation_retry_count", 0) < _max_retries(state):
        return "implement"
    return "collect"


def _after_validate_collect(state: PipelineState) -> Literal["collect", "analyze"]:
    if not state.get("validation_ok", True) and state.get("validation_retry_count", 0) < _max_retries(state):
        return "collect"
    return "analyze"


def _after_validate_analyst(state: PipelineState) -> Literal["analyze", "interpret"]:
    if not state.get("validation_ok", True) and state.get("validation_retry_count", 0) < _max_retries(state):
        return "analyze"
    return "interpret"


def _after_validate_interpret(state: PipelineState) -> Literal["interpret", "end"]:
    if not state.get("validation_ok", True) and state.get("validation_retry_count", 0) < _max_retries(state):
        return "interpret"
    return "end"


def build_graph(checkpoint_dir: str | None = None):
    """
    Build the 6-agent graph with validation loop after each agent.
    Implement step includes deploy (config.json). Collect has modes: simulated, real (real not implemented).
    """
    graph = StateGraph(PipelineState)

    graph.add_node("theory", run_theorist)
    graph.add_node("design", run_experiment_designer)
    graph.add_node("implement", run_experiment_implementer)
    graph.add_node("collect", run_collect)
    graph.add_node("analyze", run_data_analyst)
    graph.add_node("interpret", run_interpreter)

    graph.add_node("validate_theory", _validator_node(NODE_TO_AGENT_KEY["theory"]))
    graph.add_node("validate_design", _validator_node(NODE_TO_AGENT_KEY["design"]))
    graph.add_node("validate_implement", _validator_node(NODE_TO_AGENT_KEY["implement"]))
    graph.add_node("validate_collect", _validator_node(NODE_TO_AGENT_KEY["collect"]))
    graph.add_node("validate_analyst", _validator_node(NODE_TO_AGENT_KEY["analyze"]))
    graph.add_node("validate_interpret", _validator_node(NODE_TO_AGENT_KEY["interpret"]))

    graph.set_entry_point("theory")
    graph.add_edge("theory", "validate_theory")
    graph.add_conditional_edges("validate_theory", _after_validate_theory, {"theory": "theory", "design": "design"})
    graph.add_edge("design", "validate_design")
    graph.add_conditional_edges("validate_design", _after_validate_design, {"design": "design", "implement": "implement"})
    graph.add_edge("implement", "validate_implement")
    graph.add_conditional_edges(
        "validate_implement",
        _after_validate_implement,
        {"implement": "implement", "collect": "collect", "analyze": "analyze"},
    )
    graph.add_edge("collect", "validate_collect")
    graph.add_conditional_edges("validate_collect", _after_validate_collect, {"collect": "collect", "analyze": "analyze"})
    graph.add_edge("analyze", "validate_analyst")
    graph.add_conditional_edges("validate_analyst", _after_validate_analyst, {"analyze": "analyze", "interpret": "interpret"})
    graph.add_edge("interpret", "validate_interpret")
    graph.add_conditional_edges("validate_interpret", _after_validate_interpret, {"interpret": "interpret", "end": END})

    if checkpoint_dir:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            saver = SqliteSaver.from_conn_string(checkpoint_dir)
            return graph.compile(checkpointer=saver)
        except Exception:
            pass
    return graph.compile()
