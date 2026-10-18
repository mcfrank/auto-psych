"""Set-specific yes criterion on top of shared-property rule matching.

People judge by simple rule matching (yes to the degree that the target has the
arithmetic properties all the examples share), but on seeing each example set
they settle on a set-specific response criterion -- a bias toward yes or no that
varies from set to set by an inferred amount and applies to every target asked
about with that set. Sets therefore differ in overall generalization breadth
beyond what their shared properties explain.
"""
import numpy as np
import pymc as pm

_PRIMES = {n for n in range(2, 101) if all(n % d for d in range(2, int(n ** 0.5) + 1))}
_SQUARES = {k * k for k in range(1, 11)}
_POW2 = {2 ** k for k in range(0, 7)}


def _props(n):
    p = {
        "even": n % 2 == 0,
        "odd": n % 2 == 1,
        "m3": n % 3 == 0,
        "m4": n % 4 == 0,
        "m5": n % 5 == 0,
        "m10": n % 10 == 0,
        "prime": n in _PRIMES,
        "square": n in _SQUARES,
        "pow2": n in _POW2,
        "end%d" % (n % 10): True,
    }
    return {k for k, v in p.items() if v}


def _rule_match(number_set, target):
    examples = [int(x) for x in str(number_set).split(",")]
    shared = None
    for e in examples:
        shared = _props(e) if shared is None else shared & _props(e)
    if not shared:
        return 0.0
    return len(shared & _props(int(target))) / len(shared) - 0.5


def prepare_observed(rows):
    """Rule-match score per trial, an index for each distinct set, and the responses."""
    set_ids = {}
    idx, match = [], []
    for r in rows:
        s = str(r["number_set"]).strip()
        if s not in set_ids:
            set_ids[s] = len(set_ids)
        idx.append(set_ids[s])
        match.append(_rule_match(s, r["target"]))
    return {
        "rule_match": np.asarray(match, dtype="float64"),
        "set_idx": np.asarray(idx, dtype="int64"),
        "set_slots": np.zeros(max(len(set_ids), 1), dtype="float64"),
        "rating": np.asarray([int(float(r["rating"])) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    rule_match = pm.Data("rule_match", np.zeros(1, dtype="float64"))
    set_idx = pm.Data("set_idx", np.zeros(1, dtype="int64"))
    set_slots = pm.Data("set_slots", np.zeros(1, dtype="float64"))

    # Population-level criterion and rule-matching sharpness.
    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    sharpness = pm.HalfNormal("sharpness", sigma=10.0)

    # Set-specific criterion shifts (non-centred).
    criterion_sd = pm.HalfNormal("criterion_sd", sigma=1.0)
    z_set = pm.Normal("z_set", 0.0, 1.0, shape=(set_slots.shape[0],))
    bias = criterion_sd * z_set[set_idx]

    p_yes = pm.Deterministic(
        "p_yes", pm.math.sigmoid(intercept + bias + sharpness * rule_match)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
