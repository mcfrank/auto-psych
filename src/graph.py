"""LangGraph workflow: 7-agent pipeline with conditional deployer -> analyst path."""

from typing import Literal

from langgraph.graph import StateGraph, END

from src.state import PipelineState
from src.agents.theorist import run_theorist
from src.agents.experiment_designer import run_experiment_designer
from src.agents.experiment_implementer import run_experiment_implementer
from src.agents.deployer import run_deployer
from src.agents.simulated_participant import run_simulated_participant
from src.agents.data_analyst import run_data_analyst
from src.agents.interpreter import run_interpreter
from src.validation.validators import run_validation, NODE_TO_AGENT_KEY

MAX_VALIDATION_RETRIES = 3


def _validator_node(agent_key: str):
    """Return a node function that runs validation for the given agent_key."""
    def _run(state: PipelineState) -> PipelineState:
        return run_validation(state, agent_key)
    return _run


def _after_validate_theorist(state: PipelineState) -> Literal["theorist", "experiment_designer"]:
    if not state.get("validation_ok", True) and state.get("validation_retry_count", 0) < MAX_VALIDATION_RETRIES:
        return "theorist"
    return "experiment_designer"


def _after_validate_designer(state: PipelineState) -> Literal["experiment_designer", "experiment_implementer"]:
    if not state.get("validation_ok", True) and state.get("validation_retry_count", 0) < MAX_VALIDATION_RETRIES:
        return "experiment_designer"
    return "experiment_implementer"


def _after_validate_implementer(state: PipelineState) -> Literal["experiment_implementer", "deployer"]:
    if not state.get("validation_ok", True) and state.get("validation_retry_count", 0) < MAX_VALIDATION_RETRIES:
        return "experiment_implementer"
    return "deployer"


def _after_validate_deployer(state: PipelineState) -> Literal["deployer", "simulated_participant", "data_analyst"]:
    if not state.get("validation_ok", True) and state.get("validation_retry_count", 0) < MAX_VALIDATION_RETRIES:
        return "deployer"
    if state.get("mode") == "simulated_participants":
        return "simulated_participant"
    return "data_analyst"


def _after_validate_simulated_participant(state: PipelineState) -> Literal["simulated_participant", "data_analyst"]:
    if not state.get("validation_ok", True) and state.get("validation_retry_count", 0) < MAX_VALIDATION_RETRIES:
        return "simulated_participant"
    return "data_analyst"


def _after_validate_analyst(state: PipelineState) -> Literal["data_analyst", "interpreter"]:
    if not state.get("validation_ok", True) and state.get("validation_retry_count", 0) < MAX_VALIDATION_RETRIES:
        return "data_analyst"
    return "interpreter"


def _after_validate_interpreter(state: PipelineState) -> Literal["interpreter", "end"]:
    if not state.get("validation_ok", True) and state.get("validation_retry_count", 0) < MAX_VALIDATION_RETRIES:
        return "interpreter"
    return "end"


def build_graph(checkpoint_dir: str | None = None):
    """
    Build the 7-agent graph with validation loop after each agent.
    Each agent -> validate_* -> conditional: retry agent (up to 3x) or next node.
    """
    graph = StateGraph(PipelineState)

    graph.add_node("theorist", run_theorist)
    graph.add_node("experiment_designer", run_experiment_designer)
    graph.add_node("experiment_implementer", run_experiment_implementer)
    graph.add_node("deployer", run_deployer)
    graph.add_node("simulated_participant", run_simulated_participant)
    graph.add_node("data_analyst", run_data_analyst)
    graph.add_node("interpreter", run_interpreter)

    graph.add_node("validate_theorist", _validator_node(NODE_TO_AGENT_KEY["theorist"]))
    graph.add_node("validate_designer", _validator_node(NODE_TO_AGENT_KEY["experiment_designer"]))
    graph.add_node("validate_implementer", _validator_node(NODE_TO_AGENT_KEY["experiment_implementer"]))
    graph.add_node("validate_deployer", _validator_node(NODE_TO_AGENT_KEY["deployer"]))
    graph.add_node("validate_simulated_participant", _validator_node(NODE_TO_AGENT_KEY["simulated_participant"]))
    graph.add_node("validate_analyst", _validator_node(NODE_TO_AGENT_KEY["data_analyst"]))
    graph.add_node("validate_interpreter", _validator_node(NODE_TO_AGENT_KEY["interpreter"]))

    graph.set_entry_point("theorist")
    graph.add_edge("theorist", "validate_theorist")
    graph.add_conditional_edges("validate_theorist", _after_validate_theorist, {"theorist": "theorist", "experiment_designer": "experiment_designer"})
    graph.add_edge("experiment_designer", "validate_designer")
    graph.add_conditional_edges("validate_designer", _after_validate_designer, {"experiment_designer": "experiment_designer", "experiment_implementer": "experiment_implementer"})
    graph.add_edge("experiment_implementer", "validate_implementer")
    graph.add_conditional_edges("validate_implementer", _after_validate_implementer, {"experiment_implementer": "experiment_implementer", "deployer": "deployer"})
    graph.add_edge("deployer", "validate_deployer")
    graph.add_conditional_edges(
        "validate_deployer",
        _after_validate_deployer,
        {"deployer": "deployer", "simulated_participant": "simulated_participant", "data_analyst": "data_analyst"},
    )
    graph.add_edge("simulated_participant", "validate_simulated_participant")
    graph.add_conditional_edges("validate_simulated_participant", _after_validate_simulated_participant, {"simulated_participant": "simulated_participant", "data_analyst": "data_analyst"})
    graph.add_edge("data_analyst", "validate_analyst")
    graph.add_conditional_edges("validate_analyst", _after_validate_analyst, {"data_analyst": "data_analyst", "interpreter": "interpreter"})
    graph.add_edge("interpreter", "validate_interpreter")
    graph.add_conditional_edges("validate_interpreter", _after_validate_interpreter, {"interpreter": "interpreter", "end": END})

    if checkpoint_dir:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            saver = SqliteSaver.from_conn_string(checkpoint_dir)
            return graph.compile(checkpointer=saver)
        except Exception:
            pass
    return graph.compile()
