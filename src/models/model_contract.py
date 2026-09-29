"""The data contract every cognitive model must honour, checked without sampling.

A model is scored on its observed variable (ELPD-LOO, pruning, export) but used
through its ``p_left`` (the EIG design, the novelty gate, held-out
evaluation). Loading only requires one observed RV backed by a ``pm.Data``
container, so a model could be fitted to something other than the responses
(``1 - chose_left``, or the rows in another order, which leaves the ELPD total
right but misaligns the pointwise LOO values with the rows and so corrupts
``dse`` and pruning), or be scored on a probability other than the ``p_left``
it reports (a lapse applied after ``p_left``). Second audit B13, first audit
D5.

``model_contract_violation`` binds a response file and evaluates the model's
graph at a few parameter points, and returns why the model breaks the
contract, or ``None``:

- the observed data bound from the file are its ``chose_left`` column, in row
  order;
- ``p_left`` exists and has one entry per trial;
- the observed variable's log-likelihood is one term per trial, and at each
  point the probability it gives each trial's observed response equals
  Bernoulli(chose_left; p_left) within ``CONTRACT_PROBABILITY_TOLERANCE``;
- no ``pm.Potential`` depends on the responses (a likelihood term outside the
  observed variable, which LOO and ``p_left`` never see).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from src.models.data_binding import _read_csv_rows, extract_observed
from src.models.model_loading import _import_pymc, load_pymc_model
from src.models.pymc_inference import is_model_failure

# The largest |P_likelihood(observed response) - Bernoulli(observed; p_left)|
# tolerated at any trial and test point, on the probability scale. The same
# probability computed along two float64 paths (a ``logit_p`` likelihood beside
# a ``sigmoid`` p_left, say) agrees to ~1e-15, and numerical clip guards
# (``clip(p, 1e-9, 1 - 1e-9)``) move it by their width; anything that changes
# what the model predicts moves it far more (a 1% lapse moves p_left = 0.9 by
# 0.004). The probability scale, not the log scale, because a guard at 1e-9
# shifts log p by nats where p itself is ~0.
CONTRACT_PROBABILITY_TOLERANCE = 1e-5

# Test points beyond the initial point: the initial point often puts every
# parameter at a symmetric value where p_left is 0.5 on every trial, and a
# lapse or mixture around 0.5 is invisible there. Jittered by U(-1, 1) on the
# unconstrained scale, as PyMC jitters its own NUTS start points, from a fixed
# seed so the verdict is reproducible.
CONTRACT_JITTERED_POINTS = 3
CONTRACT_JITTER_SEED = 0

P_LEFT_NAME = "p_left"


def _responses_chose_left(responses_path: Path) -> np.ndarray:
    """The response file's ``chose_left`` column, as 0/1 integers in row order."""
    rows = _read_csv_rows(responses_path)
    if not rows or "chose_left" not in rows[0]:
        raise ValueError(
            f"{responses_path} has no chose_left column; the contract check needs "
            "a response file."
        )
    values = np.array([float(row["chose_left"]) for row in rows])
    if not np.all((values == 0) | (values == 1)):
        raise ValueError(f"{responses_path}: chose_left must be 0 or 1 on every row.")
    return values.astype(int)


def _describe_mismatch(bound: np.ndarray, chose_left: np.ndarray) -> str:
    differing = np.flatnonzero(bound != chose_left)
    what = (
        "chose_left in another row order"
        if np.array_equal(np.sort(bound), np.sort(chose_left))
        else "not the chose_left column"
    )
    return (
        f"the observed data the model binds from the responses are {what} "
        f"({len(differing)} of {len(chose_left)} trials differ, first at row "
        f"{differing[0]}); the observed variable must be exactly the responses' "
        "chose_left, one per trial, in row order"
    )


def _test_points(model: Any) -> List[dict]:
    initial = model.initial_point()
    rng = np.random.default_rng(CONTRACT_JITTER_SEED)
    jittered = [
        {
            key: value + rng.uniform(-1.0, 1.0, size=np.shape(value))
            for key, value in initial.items()
        }
        for _ in range(CONTRACT_JITTERED_POINTS)
    ]
    return [initial, *jittered]


def _response_potentials(model: Any, observed_value: Any) -> List[str]:
    try:
        from pytensor.graph.traversal import ancestors
    except ImportError:  # pytensor < 2.31 kept it in graph.basic
        from pytensor.graph.basic import ancestors

    response_inputs = set(ancestors([observed_value]))
    return [
        potential.name or "<unnamed>"
        for potential in model.potentials
        if response_inputs & set(ancestors([potential]))
    ]


def _check_bound_model(model: Any, chose_left: np.ndarray) -> Optional[str]:
    """The contract checks on a model whose data are already bound."""
    from pytensor.compile.function import function

    n_trials = len(chose_left)
    if len(model.observed_RVs) != 1:
        return (
            f"the model has {len(model.observed_RVs)} observed variables; it must "
            "have exactly one, the Bernoulli response"
        )
    response = model.observed_RVs[0]
    observed_value = model.rvs_to_values[response]

    bound = np.asarray(function([], observed_value)())
    if bound.shape != (n_trials,):
        return (
            f"the observed data have shape {bound.shape}, but the responses have "
            f"{n_trials} trials; the observed variable must be exactly the "
            "responses' chose_left, one per trial, in row order"
        )
    if not np.array_equal(bound, chose_left):
        return _describe_mismatch(bound, chose_left)

    potentials = _response_potentials(model, observed_value)
    if potentials:
        return (
            f"pm.Potential {', '.join(repr(p) for p in potentials)} depends on the "
            "responses: a likelihood term outside the observed variable is invisible "
            "to ELPD-LOO and to p_left. Put the whole likelihood in the one "
            "Bernoulli observed variable"
        )

    if P_LEFT_NAME not in model.named_vars:
        return (
            "the model has no variable named 'p_left'; it must define "
            "pm.Deterministic('p_left', ...), the per-trial probability of choosing "
            "the left sequence that its Bernoulli likelihood uses"
        )
    (p_left,) = model.replace_rvs_by_values([model.named_vars[P_LEFT_NAME]])
    (log_likelihood,) = model.logp(vars=[response], sum=False, jacobian=False)
    evaluate = function(
        model.value_vars, [p_left, log_likelihood], on_unused_input="ignore"
    )

    expected_side = chose_left == 1
    for i, point in enumerate(_test_points(model)):
        p, ll = (np.asarray(v, dtype=float) for v in evaluate(
            *[point[v.name] for v in model.value_vars]
        ))
        if p.shape != (n_trials,):
            return (
                f"p_left has shape {p.shape}, not one entry per trial ({n_trials} "
                "trials); it must be a vector with one probability per response row"
            )
        if ll.shape != (n_trials,):
            return (
                f"the observed variable's log-likelihood has shape {ll.shape}, not "
                f"one term per trial ({n_trials} trials); ELPD-LOO needs one "
                "Bernoulli term per response row"
            )
        finite = np.isfinite(p) & np.isfinite(ll)
        if i > 0 and not finite.all():
            continue  # a jittered point off the model's support: nothing to compare
        if not finite.all():
            return "p_left or the log-likelihood is not finite at the initial point"
        implied = np.exp(ll)
        bernoulli = np.where(expected_side, p, 1.0 - p)
        gap = np.abs(implied - bernoulli)
        worst = int(np.argmax(gap))
        if gap[worst] > CONTRACT_PROBABILITY_TOLERANCE:
            return (
                "the likelihood's probability of the observed responses disagrees "
                f"with Bernoulli(chose_left; p_left): at row {worst} "
                f"{'the initial point' if i == 0 else 'a jittered test point'} gives "
                f"p_left = {p[worst]:.6g}, so Bernoulli probability "
                f"{bernoulli[worst]:.6g}, but the likelihood gives "
                f"{implied[worst]:.6g}. p_left must be exactly the probability the "
                "Bernoulli likelihood uses (apply any lapse, mixture or bias to "
                "p_left itself)"
            )
    return None


def model_contract_violation(
    name: str, models_dir: Path, responses_path: Path
) -> Optional[str]:
    """Why ``models_dir/<name>.py`` breaks the data contract on ``responses_path``,
    or ``None`` when it honours it (see the module docstring). No sampling.

    A failure of the model's own code while binding or evaluating is a
    violation with its reason; a failure of the harness or the machine raises
    (``is_model_failure``), as in ``model_logp_is_finite``.
    """
    pm = _import_pymc()
    model_file = Path(models_dir) / f"{name}.py"
    chose_left = _responses_chose_left(responses_path)
    model = load_pymc_model(name, models_dir)
    try:
        observed = extract_observed(responses_path, model)
        with model:
            pm.set_data(observed)
        return _check_bound_model(model, chose_left)
    except Exception as e:
        if not is_model_failure(e, model_file):
            raise
        return f"the contract check raised: {type(e).__name__}: {e}"
