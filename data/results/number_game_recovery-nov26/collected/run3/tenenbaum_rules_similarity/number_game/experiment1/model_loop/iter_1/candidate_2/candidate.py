"""Hypothesis selection rather than hypothesis averaging in concept learning.

People do not average their predictions over every number concept consistent
with the examples; they commit to (approximately) the single most plausible
concept and generalize only from it. Given the same 45 rules and 5,050
intervals of 1..100, family-organised rule prior (lambda_0 for rules, inferred
family weights, uniform within family; each interval (1 - lambda_0) / 100^2)
and size-principle likelihood (90% of examples from the concept) as the
current best model, the concept a person generalizes from is chosen with
probability proportional to its posterior raised to an inferred decisiveness
exponent beta (beta > 1 approaches MAP selection), and "yes" means the chosen
concept contains the target.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

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
EXAMPLE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)


def rule_family(name):
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
N_INTERVALS = len(INTERVALS)
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]
INTERVAL_SIZE = INTERVAL_MEMBERSHIP.sum(axis=1).astype(int)
MAX_EXAMPLES = 4
# Within a set, every interval with the same size s that contains the same
# number k of the examples has the same prior and likelihood, hence the same
# tempered weight. Grouping intervals by (s, k) makes the tempered choice over
# 5,050 intervals exact at ~1/10 of the cost; prepare_observed counts, per set,
# what share of each group's intervals contains each target.
N_GROUPS = 100 * (MAX_EXAMPLES + 1)  # upper bound; prepare_observed trims to the sets' need

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules): log P(D | rule)
    group_log_lik = pm.Data("group_log_lik", np.zeros((1, N_GROUPS)))  # (sets, groups): log P(D | interval)
    group_log_count = pm.Data("group_log_count", np.zeros((1, N_GROUPS)))  # log #intervals (-1e4 if none)
    group_in_concept = pm.Data("group_in_concept", np.zeros((1, N_GROUPS, 100)))  # share containing target
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set index * 100 + target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    family_weights = pm.Dirichlet("family_weights", a=np.full(N_FAMILIES, 2.0))
    # Decisiveness: 1 = posterior averaging, large = commit to the MAP concept.
    beta = pm.LogNormal("beta", mu=0.0, sigma=1.0)

    rule_log_prior = pt.log(lambda_0) + pt.log(family_weights)[FAMILY] - np.log(FAMILY_SIZE[FAMILY])
    interval_log_prior = pt.log1p(-lambda_0) - 2 * np.log(100)
    log_rules = beta * (rule_log_prior[None, :] + rule_log_lik)
    log_groups = beta * (interval_log_prior + group_log_lik) + group_log_count
    log_choice = pt.concatenate([log_rules, log_groups], axis=1)
    choice = pt.exp(log_choice - pt.logsumexp(log_choice, axis=1, keepdims=True))  # (sets, rules + groups)
    in_concept = choice[:, :N_RULES] @ pt.as_tensor(MEMBERSHIP[:N_RULES]) + pt.batched_dot(
        choice[:, N_RULES:], group_in_concept
    )  # (sets, targets)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    n_sets = len(unique)
    rule_ll = np.zeros((n_sets, N_RULES))
    group_ll = np.zeros((n_sets, N_GROUPS))
    group_log_count = np.full((n_sets, N_GROUPS), -1e4)
    group_in = np.zeros((n_sets, N_GROUPS, 100))
    for i, s in enumerate(unique):
        cols = np.array(s) - 1
        ll = EXAMPLE_LOG_LIK[:, cols].sum(axis=1)
        rule_ll[i] = ll[:N_RULES]
        k = INTERVAL_MEMBERSHIP[:, cols].sum(axis=1).astype(int)
        key = (INTERVAL_SIZE - 1) * (MAX_EXAMPLES + 1) + k
        # Keep only this set's non-empty groups, packed into the first slots.
        keys, group, counts = np.unique(key, return_inverse=True, return_counts=True)
        group_log_count[i, : len(keys)] = np.log(counts)
        # Intervals in one group share a likelihood; take any member's.
        group_ll[i, group] = ll[N_RULES:]
        np.add.at(group_in[i], group, INTERVAL_MEMBERSHIP)
        group_in[i, : len(keys)] /= counts[:, None]
    used = int((group_log_count > -1e4).sum(axis=1).max())
    group_ll, group_log_count, group_in = group_ll[:, :used], group_log_count[:, :used], group_in[:, :used]
    return {
        "rule_log_lik": rule_ll,
        "group_log_lik": group_ll,
        "group_log_count": group_log_count,
        "group_in_concept": group_in,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
