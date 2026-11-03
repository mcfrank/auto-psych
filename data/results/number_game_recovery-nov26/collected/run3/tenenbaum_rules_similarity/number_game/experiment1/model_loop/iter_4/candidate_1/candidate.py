"""People do Bayesian concept learning over the same rules (family-organised
prior, exception rules as their own kind), magnitude ranges and odd/even-within-
a-range concepts as parity_within_range_concepts, with the size principle, but a
rule only comes to mind once a pattern has been noticed: the prior odds of
thinking of an arithmetic rule rather than a magnitude range grow (log-linearly,
at an inferred rate) with the number of examples shown. With one or two examples
people generalise by magnitude; with four examples they look for a rule, more
than the likelihood alone would make them.
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
PARITY_RANGES = [[n for n in c if n % 2 == p] for p in (1, 0) for c in INTERVALS]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS + PARITY_RANGES], dtype="float64")
EXAMPLE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / np.maximum(MEMBERSHIP.sum(axis=1, keepdims=True), 1) + (1 - ALPHA) / 100)
EXAMPLE_LOG_LIK[MEMBERSHIP.sum(axis=1) == 0] = -1e4  # empty concepts are impossible
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]
_SPAN = np.array([len(c) for c in INTERVALS] * 3) - 1
INTERVAL_SIZE_IDX = _SPAN + 100 * np.repeat([0, 1, 1], len(INTERVALS))  # span class; parity ranges share 100..199
SIZES = np.arange(1, 101, dtype="float64")
LOG_N_OF_SIZE = np.log(101.0 - SIZES)  # number of intervals of each size

def rule_family(name):
    if name.startswith("powers of 2,"):  # a rule with an arbitrary exception
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

# All intervals of one size share their prior, so prepare_observed sums each
# set's interval likelihood within every size (parameter-free) and records, per
# size, the share of that evidence from intervals containing each target.

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    size_log_evidence = pm.Data("size_log_evidence", np.zeros((1, 200)))  # (sets, sizes): log sum_j P(D|j)
    set_n_examples = pm.Data("set_n_examples", np.zeros(1))  # (sets,): number of examples minus 2.5
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # (cells,)
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))  # (cells, rules)
    cell_size_member = pm.Data("cell_size_member", np.zeros((1, 200)))  # (cells, sizes)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # (trials,)
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    kind = pm.Dirichlet("kind_weights", a=np.full(3, 2.0))  # rules, ranges, odd/even within a range
    sigma = pm.LogNormal("sigma", mu=np.log(15.0), sigma=1.0)  # Erlang scale of interval width
    rule_noticing = pm.Normal("rule_noticing", mu=0.0, sigma=1.0)  # log-odds gain for rules per extra example
    family_weights = pm.Dirichlet("family_weights", a=np.full(N_FAMILIES, 2.0))
    rule_log_prior = pt.log(family_weights)[FAMILY] - np.log(FAMILY_SIZE[FAMILY])

    log_size_prior = np.log(SIZES) - SIZES / sigma
    log_size_prior = log_size_prior - pt.logsumexp(log_size_prior)
    log_kind = pt.log(kind)[None, :] + pt.stack([rule_noticing * set_n_examples,
                                                 pt.zeros_like(set_n_examples),
                                                 pt.zeros_like(set_n_examples)], axis=1)
    log_kind = log_kind - pt.logsumexp(log_kind, axis=1, keepdims=True)  # (sets, 3)
    log_per_interval = pt.concatenate([log_size_prior - LOG_N_OF_SIZE,
                                       log_size_prior - LOG_N_OF_SIZE - np.log(2.0)])
    kind_of_size = np.repeat([1, 2], 100)

    log_rules = log_kind[:, 0:1] + rule_log_prior[None, :] + rule_log_lik
    log_sizes = log_kind[:, kind_of_size] + log_per_interval[None, :] + size_log_evidence
    log_joint = pt.concatenate([log_rules, log_sizes], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    cell_post = posterior[cell_set]
    in_concept = ((cell_post[:, :N_RULES] * cell_rule_member).sum(axis=1)
                  + (cell_post[:, N_RULES:] * cell_size_member).sum(axis=1))

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    set_log_lik = np.stack([EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique])
    n_sets = len(unique)
    size_ev = np.zeros((n_sets, 200))
    size_member = np.zeros((n_sets, 200, 100))
    for i in range(n_sets):
        ll = set_log_lik[i, N_RULES:]
        for L in range(200):
            sel = INTERVAL_SIZE_IDX == L
            ev = logsumexp(ll[sel])
            size_ev[i, L] = ev
            size_member[i, L] = np.exp(ll[sel] - ev) @ INTERVAL_MEMBERSHIP[sel]
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: i for i, c in enumerate(cells)}
    cell_s = np.array([c[0] for c in cells], dtype="int64")
    cell_t = np.array([c[1] - 1 for c in cells], dtype="int64")
    return {
        "rule_log_lik": set_log_lik[:, :N_RULES],
        "size_log_evidence": size_ev,
        "set_n_examples": np.array([len(s) - 2.5 for s in unique], dtype="float64"),
        "cell_set": cell_s,
        "cell_rule_member": RULE_MEMBERSHIP[:, cell_t].T.copy(),
        "cell_size_member": np.clip(size_member[cell_s, :, cell_t], 0.0, 1.0),
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
