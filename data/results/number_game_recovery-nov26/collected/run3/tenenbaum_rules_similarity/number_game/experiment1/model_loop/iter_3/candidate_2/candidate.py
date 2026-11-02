"""People read "would the program produce this number?" generatively: they judge
how likely the program is to produce that particular number as its next output
-- the posterior-predictive sampling probability, averaging over rules and
magnitude ranges that explain the examples, each concept spreading its output
thinly over its members (the size principle applied to the target as well) --
and answer "yes" according to how that probability compares with the most
likely next output for the set, raised to an inferred sensitivity. Targets only
large concepts contain thus get fewer "yes" answers than membership implies.

The concept learner (45 rules, family-organised rule prior with exception rules
as their own kind, Erlang prior over range width, 90% of examples from the
concept) is the same as the best model's; only the judgment differs.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

ALPHA = 0.9
NUMBERS = np.arange(1, 101)


def rule_concepts():
    rules = {
        "odd": [n for n in NUMBERS if n % 2 == 1],
        "primes": [n for n in NUMBERS if n > 1 and all(n % a for a in range(2, int(n**0.5) + 1))],
        "squares": [n * n for n in range(1, 11)],
        "cubes": [n**3 for n in range(1, 5)],
        "powers of 2, not 32": [2, 4, 8, 16, 64],
        "powers of 2, or 37": [2, 4, 8, 16, 32, 37, 64],
        "even": list(range(2, 101, 2)),
    }
    rules.update({f"multiples of {m}": list(range(m, 101, m)) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]})
    rules.update({f"powers of {m}": [m**k for k in range(1, 8) if m**k <= 100] for m in range(2, 11)})
    rules.update({f"ends in {d}": [n for n in NUMBERS if n % 10 == d] for d in range(10)})
    rules.update({f"contains digit {d}": [n for n in NUMBERS if str(d) in str(n)] for d in range(10)})
    return rules


_RULES = rule_concepts()
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
SAMPLE_PROB = ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100
EXAMPLE_LOG_LIK = np.log(SAMPLE_PROB)
RULE_SAMPLE_PROB = SAMPLE_PROB[:N_RULES]  # (rules, 100): P(next output = t | rule)
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]
INTERVAL_SIZE_IDX = (INTERVAL_MEMBERSHIP.sum(axis=1) - 1).astype("int64")
SIZES = np.arange(1, 101, dtype="float64")
LOG_N_OF_SIZE = np.log(101.0 - SIZES)


def rule_family(name):
    if name.startswith("powers of 2,"):
        return 7
    if name in ("odd", "even"):
        return 0
    if name == "primes":
        return 1
    if name in ("squares", "cubes"):
        return 2
    if name.startswith("powers"):
        return 3
    if name.startswith("multiples"):
        return 4
    if name.startswith("ends in"):
        return 5
    return 6


FAMILY = np.array([rule_family(n) for n in RULE_NAMES])
N_FAMILIES = FAMILY.max() + 1
FAMILY_SIZE = np.bincount(FAMILY)

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    size_log_evidence = pm.Data("size_log_evidence", np.zeros((1, 100)))  # (sets, sizes)
    size_sample_prob = pm.Data("size_sample_prob", np.zeros((1, 100, 100)))  # (sets, sizes, targets)
    row_set = pm.Data("row_set", np.zeros(1, dtype="int64"))
    row_target = pm.Data("row_target", np.zeros(1, dtype="int64"))  # 0-based
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    sigma = pm.LogNormal("sigma", mu=np.log(15.0), sigma=1.0)
    family_weights = pm.Dirichlet("family_weights", a=np.full(N_FAMILIES, 2.0))
    # How steeply "yes" falls as the target becomes a less likely next output.
    sensitivity = pm.HalfNormal("sensitivity", sigma=1.0)

    rule_log_prior = pt.log(family_weights)[FAMILY] - np.log(FAMILY_SIZE[FAMILY])
    log_size_prior = np.log(SIZES) - SIZES / sigma
    log_size_prior = log_size_prior - pt.logsumexp(log_size_prior)
    log_per_interval = log_size_prior - LOG_N_OF_SIZE

    log_rules = pt.log(lambda_0) + rule_log_prior[None, :] + rule_log_lik
    log_sizes = pt.log1p(-lambda_0) + log_per_interval[None, :] + size_log_evidence
    log_joint = pt.concatenate([log_rules, log_sizes], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    # P(next output = t | examples), (sets, 100)
    predictive = (pt.dot(posterior[:, :N_RULES], RULE_SAMPLE_PROB)
                  + (posterior[:, N_RULES:, None] * size_sample_prob).sum(axis=1))
    log_rel = pt.log(predictive) - pt.log(pt.max(predictive, axis=1, keepdims=True))
    p_set = pt.exp(sensitivity * log_rel)

    p_yes = pm.Deterministic("p_yes", pt.clip(p_set[row_set, row_target], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    set_log_lik = np.stack([EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique])
    n_sets = len(unique)
    size_ev = np.zeros((n_sets, 100))
    size_pred = np.zeros((n_sets, 100, 100))
    for i in range(n_sets):
        ll = set_log_lik[i, N_RULES:]
        for L in range(100):
            sel = INTERVAL_SIZE_IDX == L
            ev = logsumexp(ll[sel])
            size_ev[i, L] = ev
            member = np.clip(np.exp(ll[sel] - ev) @ INTERVAL_MEMBERSHIP[sel], 0.0, 1.0)
            size_pred[i, L] = ALPHA * member / (L + 1) + (1 - ALPHA) / 100
    return {
        "rule_log_lik": set_log_lik[:, :N_RULES],
        "size_log_evidence": size_ev,
        "size_sample_prob": size_pred,
        "row_set": np.array([index[s] for s in sets], dtype="int64"),
        "row_target": np.array([int(r["target"]) - 1 for r in rows], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
