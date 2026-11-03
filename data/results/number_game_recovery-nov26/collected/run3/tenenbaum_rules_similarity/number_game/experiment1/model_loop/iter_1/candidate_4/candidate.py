"""Refinement of `rule_family_prior_concepts`: Bayesian concept learning over
45 rules (family-organised rule prior) and all intervals of 1..100, with the
size principle, where the share of examples believed to be genuine samples of
the concept (alpha, versus uniform random numbers from 1..100) is inferred
rather than fixed at 0.9.

alpha sets how sharply each additional example narrows generalization to the
smallest fitting concept and how readily concepts that miss an example
survive. Everything else (rules, families, interval prior, "yes" = posterior
probability the concept contains the target) is unchanged.
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
SIZES = MEMBERSHIP.sum(axis=1)


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
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]
INTERVAL_SIZES = SIZES[N_RULES:].astype(int)
MAX_K = 4
# Intervals of equal size containing equally many examples share a likelihood,
# so per set they are pooled into one group per (size, examples inside) pair.
N_GROUPS = 100 * (MAX_K + 1)
GROUP_SIZE = np.repeat(np.arange(1, 101), MAX_K + 1).astype("float64")
GROUP_K = np.tile(np.arange(MAX_K + 1), 100).astype("float64")

with pm.Model() as model:
    rule_inside = pm.Data("rule_inside", np.zeros((1, N_RULES)))  # (sets, rules): examples inside
    n_examples = pm.Data("n_examples", np.ones(1))  # (sets,)
    group_count = pm.Data("group_count", np.zeros((1, N_GROUPS)))  # (sets, groups): intervals in group
    # Only the distinct (set, target) cells asked about are computed.
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # (cells,): set index
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))  # (cells, rules): rule contains target
    cell_group_cover = pm.Data("cell_group_cover", np.zeros((1, N_GROUPS)))  # (cells, groups): intervals containing target
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # (trials,): cell index
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    family_weights = pm.Dirichlet("family_weights", a=np.full(N_FAMILIES, 2.0))
    # Conservatism: the weight given to the examples' evidence (1 = Bayes).
    tau = pm.LogNormal("tau", mu=0.0, sigma=0.5)

    log_noise = np.log((1 - ALPHA) / 100)
    n_out = n_examples[:, None]

    rule_log_in = np.log(ALPHA / SIZES[:N_RULES] + (1 - ALPHA) / 100)
    rule_log_prior = pt.log(lambda_0) + pt.log(family_weights)[FAMILY] - np.log(FAMILY_SIZE[FAMILY])
    log_rules = rule_log_prior[None, :] + tau * (rule_inside * rule_log_in[None, :] + (n_out - rule_inside) * log_noise)

    group_log_in = np.log(ALPHA / GROUP_SIZE + (1 - ALPHA) / 100)
    group_log_lik = GROUP_K[None, :] * group_log_in[None, :] + (n_out - GROUP_K[None, :]) * log_noise
    # Log weight of one interval in each group (empty groups get -inf-like weight).
    log_one_interval = pt.log1p(-lambda_0) - 2 * np.log(100) + tau * group_log_lik
    log_groups = log_one_interval + pt.log(pt.maximum(group_count, 1e-300))

    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    log_norm = pt.logsumexp(log_joint, axis=1, keepdims=True)
    rule_post = pt.exp(log_rules - log_norm)
    one_interval_post = pt.exp(log_one_interval - log_norm)  # (sets, groups)
    in_concept = (rule_post[cell_set] * cell_rule_member).sum(axis=1) + (
        one_interval_post[cell_set] * cell_group_cover
    ).sum(axis=1)  # (cells,)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    count = np.zeros((len(unique), N_GROUPS))
    cover = np.zeros((len(unique), N_GROUPS, 100))
    for i, s in enumerate(unique):
        k = INTERVAL_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1).astype(int)
        group = (INTERVAL_SIZES - 1) * (MAX_K + 1) + k
        np.add.at(count[i], group, 1.0)
        np.add.at(cover[i], group, INTERVAL_MEMBERSHIP)
    cells = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    unique_cells = list(dict.fromkeys(cells))
    cell_index = {c: j for j, c in enumerate(unique_cells)}
    cell_set = np.array([c[0] for c in unique_cells], dtype="int64")
    cell_target = np.array([c[1] - 1 for c in unique_cells], dtype="int64")
    return {
        "rule_inside": np.stack([RULE_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "n_examples": np.array([float(len(s)) for s in unique]),
        "group_count": count,
        "cell_set": cell_set,
        "cell_rule_member": RULE_MEMBERSHIP[:, cell_target].T.copy(),
        "cell_group_cover": cover[cell_set, :, cell_target],
        "row_cell": np.array([cell_index[c] for c in cells], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
