"""Refinement of mixed_strong_weak_sampling_concepts: the same Bayesian concept
learning over 45 rules (family-organised prior, exception rules as their own
family) and all intervals of 1..100 with a beta prior over the share of the line
a range covers, 10% of examples stray. The single change is the level at which
strong and weak sampling mix: people believe the whole set of examples was
generated one way, either drawn at random from the concept (strong sampling, the
size principle, with inferred probability theta) or picked as numbers the
concept happens to contain without regard to its size (weak sampling), rather
than deciding this afresh for each example. So the weak-sampling floor does not
shrink with every added example: with three or four examples a broad concept
that fits them keeps a fixed share of belief instead of being squeezed out,
while one-example generalisation is nearly that of pure strong sampling.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
NUMBERS = np.arange(1, 101)
MAX_K = 5  # examples per set <= 4, so 0..4 inside


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
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
SIZES = np.arange(1, 101, dtype="float64")
LOG_N_OF_SIZE = np.log(101.0 - SIZES)  # number of intervals of each size
SHARE = (SIZES - 0.5) / 100.0  # share of the 1..100 line covered, in (0, 1)
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


def member_log_lik(strong, size):
    """Log probability of one example that the concept contains, under strong
    (drawn from the concept) or weak (any number, size-blind) sampling, + strays."""
    return np.log(ALPHA * (strong / size + (1.0 - strong) / 100.0) + (1.0 - ALPHA) / 100.0)


LOG_STRAY = np.log((1.0 - ALPHA) / 100.0)

# Intervals of one size share their prior and, given how many examples they
# contain, their likelihood; prepare_observed counts intervals per (size, k)
# and those of them containing each target (parameter-free).

with pm.Model() as model:
    set_n = pm.Data("set_n", np.zeros(1))  # (sets,) number of examples
    rule_k = pm.Data("rule_k", np.zeros((1, N_RULES)))  # (sets, rules) examples inside
    log_count = pm.Data("log_count", np.zeros((1, 100, MAX_K)))  # (sets, sizes, k) log #intervals
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    cell_member_share = pm.Data("cell_member_share", np.zeros((1, 100, MAX_K)))  # share containing target
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    mean_share = pm.Beta("mean_share", 2, 3)
    concentration = pm.LogNormal("concentration", mu=np.log(5.0), sigma=1.0)
    family_weights = pm.Dirichlet("family_weights", a=np.full(N_FAMILIES, 2.0))
    theta = pm.Beta("theta", 4.0, 1.0)  # belief the set was strongly sampled
    rule_log_prior = pt.log(family_weights)[FAMILY] - np.log(FAMILY_SIZE[FAMILY])

    log_size_prior = ((mean_share * concentration - 1.0) * np.log(SHARE)
                      + ((1.0 - mean_share) * concentration - 1.0) * np.log1p(-SHARE))
    log_size_prior = log_size_prior - pt.logsumexp(log_size_prior)
    log_per_interval = log_size_prior - LOG_N_OF_SIZE

    # Strong / weak per-example log likelihoods of a member (numpy, fixed).
    rule_in_s, rule_in_w = member_log_lik(1.0, RULE_SIZE), member_log_lik(0.0, RULE_SIZE)
    int_in_s, int_in_w = member_log_lik(1.0, SIZES), member_log_lik(0.0, SIZES)
    log_th, log_1m = pt.log(theta), pt.log1p(-theta)
    out_ll_rule = (set_n[:, None] - rule_k) * LOG_STRAY
    rule_ll = pt.logaddexp(log_th + rule_k * rule_in_s[None, :],
                           log_1m + rule_k * rule_in_w[None, :]) + out_ll_rule
    k3 = KS[None, None, :]
    int_ll = pt.logaddexp(log_th + k3 * int_in_s[None, :, None],
                          log_1m + k3 * int_in_w[None, :, None])
    int_ll = int_ll + (set_n[:, None, None] - k3) * LOG_STRAY  # (sets, sizes, k)
    int_ll = int_ll + log_count  # -inf-safe: absent cells have log_count -1e4
    size_log_evidence = pt.logsumexp(int_ll, axis=2)  # (sets, sizes)

    log_rules = pt.log(lambda_0) + rule_log_prior[None, :] + rule_ll
    log_sizes = pt.log1p(-lambda_0) + log_per_interval[None, :] + size_log_evidence
    log_joint = pt.concatenate([log_rules, log_sizes], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    cell_post = posterior[cell_set]
    # within each size, the share of evidence from intervals containing the target
    w = pt.exp(int_ll - size_log_evidence[:, :, None])[cell_set]  # evidence share of each k
    within = (w * cell_member_share).sum(axis=2)
    within = pt.clip(within, 0.0, 1.0)
    in_concept = ((cell_post[:, :N_RULES] * cell_rule_member).sum(axis=1)
                  + (cell_post[:, N_RULES:] * within).sum(axis=1))

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    n_sets = len(unique)
    lo = NUMBERS[:, None]  # interval start
    hi = NUMBERS[None, :]  # interval end
    valid = hi >= lo
    size_idx = (hi - lo)  # L - 1
    counts = np.zeros((n_sets, 100, MAX_K))
    k_of = []
    for i, s in enumerate(unique):
        ex = np.array(s)
        k = sum(((lo <= e) & (e <= hi)).astype(int) for e in ex)
        k_of.append(k)
        np.add.at(counts[i], (size_idx[valid], k[valid]), 1.0)
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: i for i, c in enumerate(cells)}
    cell_s = np.array([c[0] for c in cells], dtype="int64")
    cell_t = np.array([c[1] - 1 for c in cells], dtype="int64")
    member_count = np.zeros((len(cells), 100, MAX_K))
    for c, (si, t) in enumerate(cells):
        sel = valid & (lo <= t) & (t <= hi)
        np.add.at(member_count[c], (size_idx[sel], k_of[si][sel]), 1.0)
    log_count = np.where(counts > 0, np.log(np.maximum(counts, 1.0)), -1e4)
    rule_k = np.stack([RULE_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique])
    return {
        "set_n": np.array([len(s) for s in unique], dtype="float64"),
        "rule_k": rule_k,
        "log_count": log_count,
        "cell_set": cell_s,
        "cell_rule_member": RULE_MEMBERSHIP[:, cell_t].T.copy(),
        "cell_member_share": np.where(counts[cell_s] > 0, member_count / np.maximum(counts[cell_s], 1.0), 0.0),
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
