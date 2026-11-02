"""Refinement of range_size_misperception_concepts with one component taken
from uncertainty_shifted_criterion. People do Bayesian concept learning over 45
rules (family-organised prior, exception rules as their own family) and all
intervals of 1..100 (beta prior over the share of the line a range covers), with
the size principle (examples come from the concept with probability 0.9), and
they misjudge range sizes: a range of true width L feels like it holds L**rho
numbers (rho inferred). The single change is the yes/no criterion: the log-odds
of answering yes is the posterior log-odds that the concept contains the target,
shifted in proportion to the entropy of the posterior over concepts (each rule
and each range width, normalised to [0, 1]); a positive shift means uncertainty
aversion (more "no" when many concepts remain open), a negative one more "yes".
Because a compressed sense of range size keeps many range widths in play for
three- and four-example sets, it also raises the uncertainty that drives the
criterion shift for exactly those sets.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
NUMBERS = np.arange(1, 101)
MAX_K = 5  # examples inside a range: 0..4


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
RULE_LOG_LIK = np.log(ALPHA * RULE_MEMBERSHIP / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
SIZES = np.arange(1, 101, dtype="float64")
LOG_N_OF_SIZE = np.log(101.0 - SIZES)  # number of intervals of each size
SHARE = (SIZES - 0.5) / 100.0
KS = np.arange(MAX_K, dtype="float64")


def rule_family(name):
    if name in ("odd", "even"):
        return 0
    if name == "primes":
        return 1
    if name in ("squares", "cubes"):
        return 2
    if name.startswith("powers of 2,"):
        return 7
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

# An interval's likelihood depends only on its size L and on how many examples k
# fall inside it, so prepare_observed counts intervals per (set, L, k) and, per
# cell, the share of those intervals containing the target.

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    log_count = pm.Data("log_count", np.zeros((1, 100, MAX_K)))  # (sets, L, k): log #intervals (-inf -> -1e9)
    set_n = pm.Data("set_n", np.ones(1))  # (sets,) number of examples
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    cell_lk_member = pm.Data("cell_lk_member", np.zeros((1, 100, MAX_K)))  # share containing target
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    mean_share = pm.Beta("mean_share", 2, 3)
    concentration = pm.LogNormal("concentration", mu=np.log(5.0), sigma=1.0)
    family_weights = pm.Dirichlet("family_weights", a=np.full(N_FAMILIES, 2.0))
    rho = pm.LogNormal("rho", mu=0.0, sigma=0.3)  # felt range size = L ** rho
    rule_log_prior = pt.log(family_weights)[FAMILY] - np.log(FAMILY_SIZE[FAMILY])

    log_size_prior = ((mean_share * concentration - 1.0) * np.log(SHARE)
                      + ((1.0 - mean_share) * concentration - 1.0) * np.log1p(-SHARE))
    log_size_prior = log_size_prior - pt.logsumexp(log_size_prior)
    log_per_interval = log_size_prior - LOG_N_OF_SIZE

    log_in = pt.log(ALPHA * pt.exp(-rho * np.log(SIZES)) + (1 - ALPHA) / 100)  # (L,)
    log_out = np.log((1 - ALPHA) / 100)
    lik_lk = (KS[None, None, :] * log_in[None, :, None]
              + (set_n[:, None, None] - KS[None, None, :]) * log_out)  # (sets, L, k)
    log_joint_lk = log_count + lik_lk + pt.log1p(-lambda_0) + log_per_interval[None, :, None]

    log_rules = pt.log(lambda_0) + rule_log_prior[None, :] + rule_log_lik
    n_sets = log_rules.shape[0]
    log_joint = pt.concatenate([log_rules, log_joint_lk.reshape((n_sets, 100 * MAX_K))], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    cell_post = posterior[cell_set]
    in_concept = ((cell_post[:, :N_RULES] * cell_rule_member).sum(axis=1)
                  + (cell_post[:, N_RULES:] * cell_lk_member.reshape((cell_post.shape[0], 100 * MAX_K))).sum(axis=1))

    # How unsure the learner is which concept holds: posterior entropy over
    # concepts (rules, and range widths summed over k), as a share of its maximum.
    size_post = posterior[:, N_RULES:].reshape((n_sets, 100, MAX_K)).sum(axis=2)
    concept_post = pt.concatenate([posterior[:, :N_RULES], size_post], axis=1)
    entropy = -(concept_post * pt.log(concept_post + 1e-12)).sum(axis=1) / np.log(N_RULES + 100)
    uncertainty_shift = pm.Normal("uncertainty_shift", mu=0.0, sigma=1.0)
    p_in = pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6)
    logit_yes = pt.log(p_in) - pt.log1p(-p_in) - uncertainty_shift * (entropy[cell_set][row_cell] - 0.5)
    p_yes = pm.Deterministic("p_yes", pt.clip(pm.math.sigmoid(logit_yes), 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    n_sets = len(unique)
    lo = np.repeat(NUMBERS, 101 - NUMBERS)
    hi = np.concatenate([np.arange(m, 101) for m in NUMBERS])
    size_idx = hi - lo
    counts = np.zeros((n_sets, 100, MAX_K))
    contain = np.zeros((n_sets, 100, MAX_K, 100))  # targets 1..100
    member = ((NUMBERS[None, :] >= lo[:, None]) & (NUMBERS[None, :] <= hi[:, None])).astype("float64")
    for i, s in enumerate(unique):
        ex = np.array(s)
        k = ((ex[None, :] >= lo[:, None]) & (ex[None, :] <= hi[:, None])).sum(axis=1)
        k = np.minimum(k, MAX_K - 1)
        flat = size_idx * MAX_K + k
        np.add.at(counts[i].reshape(-1), flat, 1.0)
        c = np.zeros((100 * MAX_K, 100))
        np.add.at(c, flat, member)
        contain[i] = c.reshape(100, MAX_K, 100)
    share = contain / np.maximum(counts[..., None], 1.0)
    log_count = np.where(counts > 0, np.log(np.maximum(counts, 1.0)), -1e9)
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: i for i, c in enumerate(cells)}
    cell_s = np.array([c[0] for c in cells], dtype="int64")
    cell_t = np.array([c[1] - 1 for c in cells], dtype="int64")
    return {
        "rule_log_lik": np.stack([RULE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "log_count": log_count,
        "set_n": np.array([float(len(s)) for s in unique]),
        "cell_set": cell_s,
        "cell_rule_member": RULE_MEMBERSHIP[:, cell_t].T.copy(),
        "cell_lk_member": np.clip(share[cell_s, :, :, cell_t], 0.0, 1.0),
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
