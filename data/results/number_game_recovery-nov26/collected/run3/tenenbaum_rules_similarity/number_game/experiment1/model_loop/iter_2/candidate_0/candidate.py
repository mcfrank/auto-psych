"""Win-stay, lose-shift sequential hypothesis testing over number concepts.

People read the examples in order and hold one concept at a time (one of 45
rules or an interval of 1..100; rule share lambda_0, family-organised rule
prior with inferred family weights, Erlang prior over interval width with
inferred scale, size-principle likelihood with 10% random examples -- the
prior and likelihood of the best model). After the first example they hold a
concept drawn from the posterior given that example. Each later example that
the held concept contains leaves it in place (win-stay, no further narrowing);
an example it does not contain makes them draw a new concept from the
posterior given all examples seen so far (lose-shift). "Yes" = probability
that the concept held after the last example contains the target. Over
people, the held-concept distribution evolves exactly as
q_k = q_{k-1} * contains(x_k) + [1 - sum q_{k-1} * contains(x_k)] * post_k.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
NUMBERS = np.arange(1, 101)
MAX_N = 4


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
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
INTERVAL_MEMBER = MEMBERSHIP[N_RULES:]
INTERVAL_SIZE = INTERVAL_MEMBER.sum(axis=1).astype(int)
SIZES = np.arange(1, 101, dtype="float64")
LOG_N_OF_SIZE = np.log(101.0 - SIZES)


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

# Within one set, intervals of the same size containing the same examples share
# prior, likelihood and stay/shift behaviour at every step, so prepare_observed
# pools them into groups (count, size, mean target membership). Exact.

with pm.Model() as model:
    rule_cum_log_lik = pm.Data("rule_cum_log_lik", np.zeros((1, MAX_N, N_RULES)))  # log P(x_1..x_k | rule)
    group_cum_log_lik = pm.Data("group_cum_log_lik", np.zeros((1, MAX_N, 1)))
    rule_contains = pm.Data("rule_contains", np.ones((1, MAX_N, N_RULES)))  # rule contains x_k
    group_contains = pm.Data("group_contains", np.ones((1, MAX_N, 1)))
    group_log_count = pm.Data("group_log_count", np.zeros((1, 1)))  # -1e4 for padding
    group_size_idx = pm.Data("group_size_idx", np.zeros((1, 1), dtype="int64"))  # size - 1
    group_in_concept = pm.Data("group_in_concept", np.zeros((1, 1, 100)))
    present = pm.Data("present", np.ones((1, MAX_N)))  # 1 if position k exists
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set index * 100 + target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    sigma = pm.LogNormal("sigma", mu=np.log(15.0), sigma=1.0)
    family_weights = pm.Dirichlet("family_weights", a=np.full(N_FAMILIES, 2.0))

    rule_log_prior = pt.log(lambda_0) + pt.log(family_weights)[FAMILY] - np.log(FAMILY_SIZE[FAMILY])
    log_size_prior = np.log(SIZES) - SIZES / sigma
    log_size_prior = log_size_prior - pt.logsumexp(log_size_prior)
    log_per_interval = pt.log1p(-lambda_0) + log_size_prior - LOG_N_OF_SIZE
    group_log_prior = log_per_interval[group_size_idx] + group_log_count  # (sets, groups)

    log_prior = pt.concatenate([pt.broadcast_to(rule_log_prior[None, :], (group_log_prior.shape[0], N_RULES)),
                                group_log_prior], axis=1)  # (sets, H)
    cum_ll = pt.concatenate([rule_cum_log_lik, group_cum_log_lik], axis=2)  # (sets, K, H)
    contains = pt.concatenate([rule_contains, group_contains], axis=2)
    log_joint = log_prior[:, None, :] + cum_ll
    post = pt.exp(log_joint - pt.logsumexp(log_joint, axis=2, keepdims=True))  # (sets, K, H)

    q = post[:, 0]
    for k in range(1, MAX_N):
        stay = q * contains[:, k]
        shift = 1.0 - stay.sum(axis=1, keepdims=True)
        q_new = stay + shift * post[:, k]
        act = present[:, k][:, None]
        q = act * q_new + (1.0 - act) * q

    in_concept = q[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP) + (
        q[:, N_RULES:, None] * group_in_concept
    ).sum(axis=1)  # (sets, targets)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    groups = []
    for s in unique:
        contains = INTERVAL_MEMBER[:, np.array(s) - 1]
        keys = np.concatenate([INTERVAL_SIZE[:, None], contains], axis=1)
        _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
        inverse = np.ravel(inverse)
        counts = np.bincount(inverse)
        member_sum = np.zeros((len(first), 100))
        np.add.at(member_sum, inverse, INTERVAL_MEMBER)
        groups.append((first, counts, member_sum / counts[:, None]))
    n_groups = max(len(g[0]) for g in groups)
    n_sets = len(unique)
    rule_ll = np.zeros((n_sets, MAX_N, N_RULES))
    group_ll = np.zeros((n_sets, MAX_N, n_groups))
    rule_c = np.ones((n_sets, MAX_N, N_RULES))
    group_c = np.ones((n_sets, MAX_N, n_groups))
    log_count = np.full((n_sets, n_groups), -1e4)
    size_idx = np.zeros((n_sets, n_groups), dtype="int64")
    in_concept = np.zeros((n_sets, n_groups, 100))
    present = np.zeros((n_sets, MAX_N))
    for i, (s, (first, counts, share)) in enumerate(zip(unique, groups)):
        g = len(first)
        log_count[i, :g] = np.log(counts)
        size_idx[i, :g] = INTERVAL_SIZE[first] - 1
        in_concept[i, :g] = share
        r_acc = np.zeros(N_RULES)
        g_acc = np.zeros(g)
        for k, x in enumerate(s):
            r_acc = r_acc + EXAMPLE_LOG_LIK[:N_RULES, x - 1]
            g_acc = g_acc + EXAMPLE_LOG_LIK[N_RULES + first, x - 1]
            rule_ll[i, k] = r_acc
            group_ll[i, k, :g] = g_acc
            rule_c[i, k] = RULE_MEMBERSHIP[:, x - 1]
            group_c[i, k, :g] = INTERVAL_MEMBER[first, x - 1]
            present[i, k] = 1.0
    return {
        "rule_cum_log_lik": rule_ll,
        "group_cum_log_lik": group_ll,
        "rule_contains": rule_c,
        "group_contains": group_c,
        "group_log_count": log_count,
        "group_size_idx": size_idx,
        "group_in_concept": in_concept,
        "present": present,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
