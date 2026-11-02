"""Win-stay, lose-shift concept learning over rules and intervals.

People do not hold a full posterior over concepts; they keep a single running
guess and revise it example by example. After the first example they pick a
concept (one of the 45 rules or an interval of 1-100) in proportion to its
Bayesian posterior given that example (independently weighted rule prior,
lambda_0 share for rules, size principle with 10% random examples). Each later
example that the current guess contains leaves it unchanged; an example it does
not contain makes them re-pick from the posterior given all examples so far.
"Yes" is the probability that the final guess contains the target, so example
order matters and the size principle only tightens generalisation on a switch.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
NUMBERS = np.arange(1, 101)
MAX_LEN = 4


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
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_EXAMPLE_LOG_LIK = np.log(ALPHA * RULE_MEMBERSHIP / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

INTERVAL_LOG_PRIOR = -2 * np.log(100)

with pm.Model() as model:
    # Positions in presented order, zero-padded for shorter sets.
    pos_log_lik = pm.Data("pos_log_lik", np.zeros((1, MAX_LEN, N_RULES + 1)))  # (sets, positions, concepts)
    pos_in = pm.Data("pos_in", np.zeros((1, MAX_LEN, N_RULES + 1)))  # concept contains the example at that position
    pos_valid = pm.Data("pos_valid", np.zeros((1, MAX_LEN)))  # position exists in the set
    group_log_size = pm.Data("group_log_size", np.zeros((1, 1)))  # (sets, groups); padding very negative
    cell_set = pm.Data("cell_set", np.zeros((1, 1)))  # (cells, sets): one-hot
    cell_in = pm.Data("cell_in", np.zeros((1, N_RULES + 1)))  # (cells, concepts): share containing the target
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    set_weights = pm.Dirichlet("set_weights", a=SET_CONCENTRATION)
    zero_split = pm.Beta("zero_split", 2, 2)
    shared = set_weights[END_0]
    rule_weights = pm.Deterministic("rule_weights", pt.concatenate([
        set_weights[:END_0], (zero_split * shared)[None], set_weights[END_0 + 1:CONTAINS_0],
        ((1 - zero_split) * shared)[None], set_weights[CONTAINS_0:],
    ]))

    log_prior = pt.concatenate([
        pt.log(lambda_0) + pt.log(rule_weights)[None, :] + pt.zeros((group_log_size.shape[0], N_RULES)),
        pt.log1p(-lambda_0) + INTERVAL_LOG_PRIOR + group_log_size,
    ], axis=1)  # (sets, concepts)

    def posterior_upto(k):
        lj = log_prior + pos_log_lik[:, :k + 1, :].sum(axis=1)
        return pt.exp(lj - pt.logsumexp(lj, axis=1, keepdims=True))

    guess = posterior_upto(0)
    for k in range(1, MAX_LEN):
        inside = pos_in[:, k, :]
        stay = guess * inside
        switch = (guess * (1 - inside)).sum(axis=1, keepdims=True)
        revised = stay + switch * posterior_upto(k)
        valid = pos_valid[:, k][:, None]
        guess = valid * revised + (1 - valid) * guess

    in_concept = ((cell_set @ guess) * cell_in).sum(axis=1)
    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def _interval_groups(examples):
    """Group the intervals of 1..100 by (length, which examples they contain)."""
    groups = {}
    for m in range(1, 101):
        for n in range(m, 101):
            key = (n - m + 1, tuple(m <= x <= n for x in examples))
            g = groups.setdefault(key, [0, np.zeros(100)])
            g[0] += 1
            g[1][m - 1:n] += 1
    return groups


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    grouped = [_interval_groups(s) for s in unique]
    n_concepts = N_RULES + max(len(g) for g in grouped)
    pos_ll = np.zeros((len(unique), MAX_LEN, n_concepts))
    pos_in = np.zeros((len(unique), MAX_LEN, n_concepts))
    valid = np.zeros((len(unique), MAX_LEN))
    log_size = np.full((len(unique), n_concepts - N_RULES), -1e4)
    in_concept = np.zeros((len(unique), n_concepts, 100))
    for i, (s, groups) in enumerate(zip(unique, grouped)):
        in_concept[i, :N_RULES] = RULE_MEMBERSHIP
        for k, x in enumerate(s):
            valid[i, k] = 1.0
            pos_ll[i, k, :N_RULES] = RULE_EXAMPLE_LOG_LIK[:, x - 1]
            pos_in[i, k, :N_RULES] = RULE_MEMBERSHIP[:, x - 1]
        for j, ((length, contains), (count, covered)) in enumerate(groups.items()):
            log_size[i, j] = np.log(count)
            in_concept[i, N_RULES + j] = covered / count
            for k, inside in enumerate(contains):
                pos_in[i, k, N_RULES + j] = float(inside)
                pos_ll[i, k, N_RULES + j] = np.log(ALPHA * inside / length + (1 - ALPHA) / 100)
    trial_cells = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(trial_cells))
    cell_index = {c: i for i, c in enumerate(cells)}
    return {
        "pos_log_lik": pos_ll,
        "pos_in": pos_in,
        "pos_valid": valid,
        "group_log_size": log_size,
        "cell_set": np.eye(len(unique))[[i for i, _ in cells]],
        "cell_in": np.array([in_concept[i, :, t - 1] for i, t in cells]),
        "row_cell": np.array([cell_index[c] for c in trial_cells], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
