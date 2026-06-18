"""Pydantic schema for the run-explorer payloads.

These models are the contract between :mod:`src.viewer.scan` (which reads the
data tree) and the frontend (which renders the JSON). Every field maps to a
concrete artifact on disk. Stages that never ran are represented as empty
collections or ``None`` — a partial run is valid data, not an error — but
corrupt artifacts raise during scanning.
"""

from __future__ import annotations

from pydantic import BaseModel


class CognitiveModel(BaseModel):
    """One cognitive model: a seed from the theory stage or an inner-loop candidate."""

    name: str
    rationale: str | None = None
    hypothesis: str | None = None
    code: str = ""
    origin: str = "seed"  # "seed" | "candidate" | "best"


class TheoryStage(BaseModel):
    report_md: str | None = None
    models: list[CognitiveModel] = []


class DesignStage(BaseModel):
    rationale_md: str | None = None
    n_stimuli: int | None = None
    n_candidates: int | None = None
    stimuli: list[dict] = []


class ExperimentStage(BaseModel):
    config: dict | None = None
    has_index_html: bool = False
    experiment_url: str | None = None


class DataSummary(BaseModel):
    n_rows: int
    n_participants: int
    columns: list[str] = []
    p_chose_left: float | None = None
    rows_preview: list[dict] = []


class Candidate(BaseModel):
    """One proposed model in one iteration of the inner model loop."""

    iteration: int
    index: int
    name: str  # e.g. "iter0_candidate0"
    hypothesis: str | None = None
    brief: str | None = None
    code: str | None = None
    posterior: dict | None = None
    transcript: str | None = None


class CritiqueStat(BaseModel):
    """One proposed posterior-predictive test statistic and its PPC result."""

    name: str
    description: str | None = None
    code: str | None = None
    t_observed: float | None = None
    null_mean: float | None = None
    null_std: float | None = None
    z_score: float | None = None
    p_value: float | None = None
    p_value_adjusted: float | None = None
    significant: bool | None = None
    error: str | None = None
    has_result: bool = False  # True once a ppc_results.json entry was matched in


class CritiqueRound(BaseModel):
    # iteration N for a critique run before model-loop round N; None for an
    # experiment-level critique (no model loop).
    iteration: int | None = None
    context_md: str | None = None
    model: str | None = None  # incumbent model the PPC critiqued
    n_significant: int | None = None
    n_replicates: int | None = None
    significance_alpha: float | None = None
    stats: list[CritiqueStat] = []


class TrajectoryStep(BaseModel):
    step: int
    iteration: int | None = None
    best_model: str
    posteriors: dict[str, float] = {}
    elpd_loo: dict[str, float] = {}


class ModelLoopStage(BaseModel):
    report_md: str | None = None
    trajectory: list[TrajectoryStep] = []
    final_posterior: dict | None = None
    candidates: list[Candidate] = []


class Experiment(BaseModel):
    project: str  # the run path this experiment belongs to
    name: str
    theory: TheoryStage = TheoryStage()
    design: DesignStage = DesignStage()
    experiment: ExperimentStage = ExperimentStage()
    data: DataSummary | None = None
    model_loop: ModelLoopStage | None = None
    critiques: list[CritiqueRound] = []
    best_model: str | None = None


# ── runs ─────────────────────────────────────────────────────────────────────
# A *run* is one execution of the outer loop: a directory (anywhere under the
# data root) that holds one or more experiments, or a single bare model loop.
# It is identified by its path relative to the data root, so the same structure
# is presented identically wherever it lives in the tree.
class RunRef(BaseModel):
    """A run discovered by the browser."""

    path: str  # relative to the data root, e.g. "outer_loop/subjective_randomness"
    label: str  # last path component
    kind: str  # "experiments" | "loop"
    n_experiments: int = 1


class RunExperimentRef(BaseModel):
    """One experiment (or bare loop) inside a run."""

    unit: str  # subdir name, or "." for the run directory itself
    name: str  # display label
    kind: str  # "experiment" | "loop"
    best_model: str | None = None
    n_candidates: int | None = None


class Run(BaseModel):
    path: str
    label: str
    figures: list[str] = []  # run-level analysis figures, relative to the run dir
    experiments: list[RunExperimentRef] = []


class RunIndex(BaseModel):
    runs: list[RunRef] = []
