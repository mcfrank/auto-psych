"""Refinement of `two_cluster_range_concepts`, with one component taken from
`kind_uncertainty_shifted_criterion`. People do Bayesian concept learning over 45
rules (family-organised prior, exception rules as their own family), single
magnitude ranges and two-cluster magnitude concepts (the union of two disjoint
ranges, each holding an example), with a beta prior over the share of the
1..100 line a magnitude concept covers and the size principle (examples come
from the concept with probability 0.9). The one change is the yes/no
criterion: the log-odds of answering yes is the posterior log-odds of
membership shifted in proportion to the normalised entropy of the posterior
over concept kinds (each rule family, single range, two clusters), so sets
that leave open whether the examples form one stretch, two groups or a rule
move answers toward "no" (or "yes").
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

ALPHA = 0.9
NUMBERS = np.arange(1, 101)
NEG = -1e4  # log-evidence for a size no consistent two-cluster concept has


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
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]
INTERVAL_SIZE_IDX = (INTERVAL_MEMBERSHIP.sum(axis=1) - 1).astype("int64")  # L - 1
SIZES = np.arange(1, 101, dtype="float64")
LOG_N_OF_SIZE = np.log(101.0 - SIZES)  # number of intervals of each size
SHARE = (SIZES - 0.5) / 100.0  # share of the 1..100 line covered, in (0, 1)
# Number of unions of two disjoint, non-adjacent ranges with total size S:
# (S - 1) splits of S into two lengths times C(101 - S, 2) placements.
N_PAIRS = (SIZES - 1) * (101 - SIZES) * (100 - SIZES) / 2
PAIR_OK = N_PAIRS > 0
LOG_N_PAIRS = np.log(np.where(PAIR_OK, N_PAIRS, 1.0))


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


def two_cluster_counts(examples):
    """Per total size S (index S-1): how many two-range unions contain every
    example with each range holding at least one, and how many of those also
    contain each target 1..100."""
    xs = sorted(set(examples))
    count = np.zeros(100)
    member = np.zeros((100, 102))  # difference array over positions 1..101
    for k in range(1, len(xs)):
        lo, hi = xs[k - 1], xs[k]  # gap lies strictly between these examples
        if hi - lo < 3:
            continue
        a = np.arange(1, xs[0] + 1)
        d = np.arange(xs[-1], 101)
        b = np.arange(lo, hi - 1)
        c = np.arange(lo + 2, hi + 1)
        bb, cc = np.meshgrid(b, c, indexing="ij")
        ok = cc >= bb + 2
        bb, cc = bb[ok], cc[ok]
        A, B, C, Dd = np.meshgrid(a, np.arange(len(bb)), np.arange(1), d, indexing="ij")
        A, Dd = A.ravel(), Dd.ravel()
        Bv, Cv = bb[B.ravel()], cc[B.ravel()]
        s = (Bv - A + 1) + (Dd - Cv + 1) - 1
        np.add.at(count, s, 1.0)
        np.add.at(member, (s, A), 1.0)
        np.add.at(member, (s, Bv + 1), -1.0)
        np.add.at(member, (s, Cv), 1.0)
        np.add.at(member, (s, Dd + 1), -1.0)
    member = np.cumsum(member, axis=1)[:, 1:101]
    return count, member


with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    size_log_evidence = pm.Data("size_log_evidence", np.zeros((1, 100)))  # (sets, sizes)
    pair_log_evidence = pm.Data("pair_log_evidence", np.zeros((1, 100)))  # (sets, total sizes)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # (cells,)
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))  # (cells, rules)
    cell_size_member = pm.Data("cell_size_member", np.zeros((1, 100)))  # (cells, sizes)
    cell_pair_member = pm.Data("cell_pair_member", np.zeros((1, 100)))  # (cells, total sizes)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # (trials,)
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    mean_share = pm.Beta("mean_share", 2, 3)  # mean share of 1..100 a magnitude concept covers
    concentration = pm.LogNormal("concentration", mu=np.log(5.0), sigma=1.0)
    family_weights = pm.Dirichlet("family_weights", a=np.full(N_FAMILIES, 2.0))
    # Share of magnitude belief given to two separate clusters rather than one range.
    two_cluster_share = pm.Beta("two_cluster_share", 1, 9)
    rule_log_prior = pt.log(family_weights)[FAMILY] - np.log(FAMILY_SIZE[FAMILY])

    log_size_prior = ((mean_share * concentration - 1.0) * np.log(SHARE)
                      + ((1.0 - mean_share) * concentration - 1.0) * np.log1p(-SHARE))
    log_size_prior = log_size_prior - pt.logsumexp(log_size_prior)
    log_per_interval = log_size_prior - LOG_N_OF_SIZE
    log_pair_size_prior = pt.where(PAIR_OK, log_size_prior, -np.inf)
    log_pair_size_prior = log_pair_size_prior - pt.logsumexp(log_pair_size_prior)
    log_per_pair = pt.where(PAIR_OK, log_pair_size_prior - LOG_N_PAIRS, NEG)

    log_magnitude = pt.log1p(-lambda_0)
    log_rules = pt.log(lambda_0) + rule_log_prior[None, :] + rule_log_lik
    log_sizes = log_magnitude + pt.log1p(-two_cluster_share) + log_per_interval[None, :] + size_log_evidence
    log_pairs = log_magnitude + pt.log(two_cluster_share) + log_per_pair[None, :] + pair_log_evidence
    log_joint = pt.concatenate([log_rules, log_sizes, log_pairs], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    cell_post = posterior[cell_set]
    in_concept = ((cell_post[:, :N_RULES] * cell_rule_member).sum(axis=1)
                  + (cell_post[:, N_RULES:N_RULES + 100] * cell_size_member).sum(axis=1)
                  + (cell_post[:, N_RULES + 100:] * cell_pair_member).sum(axis=1))

    # How unsure the learner is which KIND of program it faces: entropy of the
    # posterior over kinds (each rule family, single ranges, two clusters), as
    # a share of its maximum.
    kind_post = pt.concatenate(
        [pt.stack([posterior[:, :N_RULES][:, FAMILY == f].sum(axis=1) for f in range(N_FAMILIES)], axis=1),
         posterior[:, N_RULES:N_RULES + 100].sum(axis=1, keepdims=True),
         posterior[:, N_RULES + 100:].sum(axis=1, keepdims=True)], axis=1)
    entropy = -(kind_post * pt.log(kind_post + 1e-12)).sum(axis=1) / np.log(N_FAMILIES + 2)
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
    set_log_lik = np.stack([EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique])
    n_sets = len(unique)
    size_ev = np.zeros((n_sets, 100))
    size_member = np.zeros((n_sets, 100, 100))
    pair_ev = np.full((n_sets, 100), NEG)
    pair_member = np.zeros((n_sets, 100, 100))
    for i, s in enumerate(unique):
        ll = set_log_lik[i, N_RULES:]
        for L in range(100):
            sel = INTERVAL_SIZE_IDX == L
            ev = logsumexp(ll[sel])
            size_ev[i, L] = ev
            size_member[i, L] = np.exp(ll[sel] - ev) @ INTERVAL_MEMBERSHIP[sel]
        # Two-cluster concepts contain every example, so each has likelihood
        # (ALPHA / S + (1 - ALPHA) / 100) ** n, depending only on its size S.
        count, member = two_cluster_counts(s)
        has = count > 0
        pair_ev[i, has] = np.log(count[has]) + len(s) * np.log(ALPHA / SIZES[has] + (1 - ALPHA) / 100)
        pair_member[i, has] = member[has] / count[has, None]
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: i for i, c in enumerate(cells)}
    cell_s = np.array([c[0] for c in cells], dtype="int64")
    cell_t = np.array([c[1] - 1 for c in cells], dtype="int64")
    return {
        "rule_log_lik": set_log_lik[:, :N_RULES],
        "size_log_evidence": size_ev,
        "pair_log_evidence": pair_ev,
        "cell_set": cell_s,
        "cell_rule_member": RULE_MEMBERSHIP[:, cell_t].T.copy(),
        "cell_size_member": np.clip(size_member[cell_s, :, cell_t], 0.0, 1.0),
        "cell_pair_member": np.clip(pair_member[cell_s, :, cell_t], 0.0, 1.0),
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
