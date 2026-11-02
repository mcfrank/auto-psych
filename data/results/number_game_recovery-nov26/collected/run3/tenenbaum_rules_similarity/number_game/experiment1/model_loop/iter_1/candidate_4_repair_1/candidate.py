"""Refinement of `rule_family_prior_concepts`: magnitude ranges on a compressed
mental number line.

People do the same Bayesian concept learning as the incumbent -- 45 rules with
a family-organised prior, ranges (intervals) of 1..100, the size principle
(each example from the concept with probability 0.9, uniform on 1..100
otherwise) and "yes" = the posterior probability that the concept contains the
target. The one change is the prior over ranges: people place a range's two
endpoints uniformly on a compressed mental number line, position
log(x + c) with an inferred offset c (large c = the incumbent's linear line,
small c = logarithmic), so a range among small numbers is a priori more likely
than an equally long range among large numbers, where numbers are crowded
together on the mental line.
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


_RULES = rule_concepts()
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")  # (rules, 100)
RULE_LOG_IN = np.log(ALPHA / RULE_MEMBERSHIP.sum(axis=1) + (1 - ALPHA) / 100)
FAMILY = np.array([rule_family(n) for n in RULE_NAMES])
N_FAMILIES = FAMILY.max() + 1
FAMILY_SIZE = np.bincount(FAMILY)

INT_LO = np.array([m for m in NUMBERS for n in range(m, 101)])
INT_HI = np.array([n for m in NUMBERS for n in range(m, 101)])
INTERVAL_MEMBERSHIP = ((NUMBERS[None, :] >= INT_LO[:, None]) & (NUMBERS[None, :] <= INT_HI[:, None])).astype("float64")
INTERVAL_LOG_IN = np.log(ALPHA / (INT_HI - INT_LO + 1) + (1 - ALPHA) / 100)
LOG_NOISE = np.log((1 - ALPHA) / 100)
MAX_MISS = 1  # ranges missing two or more examples have negligible posterior

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules): log P(D | rule)
    # Ranges kept for each set (flattened over sets): which set, which range, log P(D | range).
    range_set = pm.Data("range_set", np.zeros(1, dtype="int64"))
    range_id = pm.Data("range_id", np.zeros(1, dtype="int64"))
    range_log_lik = pm.Data("range_log_lik", np.zeros(1))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # (cells,)
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))  # (cells, rules)
    # Sparse (cell, kept range) pairs where the range contains the cell's target.
    cover_cell = pm.Data("cover_cell", np.zeros(1, dtype="int64"))
    cover_range = pm.Data("cover_range", np.zeros(1, dtype="int64"))  # index into the kept ranges
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # (trials,)
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    family_weights = pm.Dirichlet("family_weights", a=np.full(N_FAMILIES, 2.0))
    # Offset of the mental number line log(x + c): small c = strong compression.
    log_c = pm.Normal("log_c", mu=np.log(10.0), sigma=1.5)
    c = pt.exp(log_c)

    # Each endpoint falls uniformly on the mental line; integer x owns the
    # stretch between x - 1/2 and x + 1/2. Prior of a range = product of its
    # endpoints' stretches, normalised over all ranges lo <= hi.
    edges = pt.log(np.arange(0.5, 101.0, 1.0) + c)
    width = edges[1:] - edges[:-1]  # (100,)
    log_w = pt.log(width)
    cum = pt.cumsum(width)  # sum_{lo <= hi} width[lo]
    log_total = pt.log(pt.sum(width * cum))
    log_range_prior = log_w[INT_LO - 1] + log_w[INT_HI - 1] - log_total  # (ranges,)

    rule_log_prior = pt.log(family_weights)[FAMILY] - np.log(FAMILY_SIZE[FAMILY])
    log_rules = pt.log(lambda_0) + rule_log_prior[None, :] + rule_log_lik  # (sets, rules)
    log_ranges = pt.log1p(-lambda_0) + log_range_prior[range_id] + range_log_lik  # (kept,)

    # Normaliser per set: rules plus that set's kept ranges.
    range_max = pt.max(log_ranges)
    range_mass = pt.zeros(rule_log_lik.shape[0])
    range_mass = pt.inc_subtensor(range_mass[range_set], pt.exp(log_ranges - range_max))
    rule_max = pt.max(log_rules, axis=1)
    log_norm = pt.logaddexp(
        rule_max + pt.log(pt.exp(log_rules - rule_max[:, None]).sum(axis=1)),
        range_max + pt.log(range_mass + 1e-300),
    )  # (sets,)

    rule_post = pt.exp(log_rules - log_norm[:, None])
    range_post = pt.exp(log_ranges - log_norm[range_set])
    in_concept = (rule_post[cell_set] * cell_rule_member).sum(axis=1)
    in_concept = pt.inc_subtensor(in_concept[cover_cell], range_post[cover_range])

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    cells = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    unique_cells = list(dict.fromkeys(cells))
    cell_index = {c: j for j, c in enumerate(unique_cells)}
    cell_set = np.array([c[0] for c in unique_cells], dtype="int64")
    cell_target = np.array([c[1] - 1 for c in unique_cells], dtype="int64")

    r_set, r_id, r_ll, cov_cell, cov_range = [], [], [], [], []
    offset = 0
    for i, s in enumerate(unique):
        ex = np.array(s) - 1
        k = INTERVAL_MEMBERSHIP[:, ex].sum(axis=1)
        keep = np.nonzero(len(s) - k <= MAX_MISS)[0]
        r_set.append(np.full(len(keep), i))
        r_id.append(keep)
        r_ll.append(k[keep] * INTERVAL_LOG_IN[keep] + (len(s) - k[keep]) * LOG_NOISE)
        for j in np.nonzero(cell_set == i)[0]:
            hit = np.nonzero(INTERVAL_MEMBERSHIP[keep, cell_target[j]])[0]
            cov_cell.append(np.full(len(hit), j))
            cov_range.append(offset + hit)
        offset += len(keep)
    return {
        "rule_log_lik": np.stack(
            [RULE_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) * (RULE_LOG_IN - LOG_NOISE) + len(s) * LOG_NOISE
             for s in unique]
        ),
        "range_set": np.concatenate(r_set).astype("int64"),
        "range_id": np.concatenate(r_id).astype("int64"),
        "range_log_lik": np.concatenate(r_ll).astype("float64"),
        "cell_set": cell_set,
        "cell_rule_member": RULE_MEMBERSHIP[:, cell_target].T.copy(),
        "cover_cell": np.concatenate(cov_cell).astype("int64"),
        "cover_range": np.concatenate(cov_range).astype("int64"),
        "row_cell": np.array([cell_index[c] for c in cells], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
