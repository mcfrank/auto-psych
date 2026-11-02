"""Refinement of narrow_range_specificity_bias: Bayesian concept learning
over the same 45 rules and every interval of 1..100, with examples taken to be
distinct draws from the concept's members not yet shown (strong sampling
without replacement, 10% of examples drawn uniformly), and a response lapse.
Within the intervals (share 1 - lambda_0) each interval's prior is
proportional to width ** -g. The one change (the component taken from
rule_breadth_prior_concepts): the rules no longer share an equal prior; each
rule's prior is proportional to its size ** -h, a breadth bias among rules
with its own strength, separate from the specificity bias on ranges -- two
distinct specificity habits, one for arithmetic/digit rules and one for
stretches of the number line.
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
    return list(rules.values())


RULES = rule_concepts()
N_RULES = len(RULES)
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
CONCEPT_SIZE = MEMBERSHIP.sum(axis=1)
INTERVAL_SIZE = np.array([len(c) for c in INTERVALS])
SIZES = np.arange(1, 101)
LOG_SIZE_C = np.log(SIZES) - np.log(SIZES).mean()  # centred so g does not trade off with lambda_0
LOG_COUNT_PER_SIZE = np.log(101 - SIZES)  # number of intervals of each width
LOG_RULE_SIZE = np.log(CONCEPT_SIZE[:N_RULES])
LOG_RULE_SIZE_C = LOG_RULE_SIZE - LOG_RULE_SIZE.mean()


def set_log_lik(examples):
    """log P(examples in order | h) for every hypothesis, sampling members without replacement."""
    total = np.zeros(len(MEMBERSHIP))
    shown = np.zeros(len(MEMBERSHIP))
    for x in examples:
        in_h = MEMBERSHIP[:, x - 1]
        remaining = np.maximum(CONCEPT_SIZE - shown, 1.0)
        total += np.log(ALPHA * in_h / remaining + (1 - ALPHA) / 100)
        shown += in_h
    return total


# Rules are kept separately (45 of them); intervals of one width share a prior, so they collapse per set and
# width. Only the (set, target) cells actually asked about are kept.

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules): log P(D | r)
    size_log_evidence = pm.Data("size_log_evidence", np.zeros((1, 100)))  # (sets, widths): mean log-evidence
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # (cells,): set index of each cell
    cell_rule_in = pm.Data("cell_rule_in", np.zeros((1, N_RULES)))  # (cells, rules): target in rule
    cell_size_in = pm.Data("cell_size_in", np.zeros((1, 100)))  # (cells, widths)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # cell index of each trial
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    g = pm.Normal("g", mu=0.0, sigma=1.0)  # Occam bias on range width
    h = pm.Normal("h", mu=0.0, sigma=1.0)  # breadth bias among rules
    log_rule_prior = -h * LOG_RULE_SIZE_C
    log_rule_prior = log_rule_prior - pt.logsumexp(log_rule_prior) + pt.log(lambda_0)
    log_size_prior = -g * LOG_SIZE_C + LOG_COUNT_PER_SIZE
    log_size_prior = log_size_prior - pt.logsumexp(log_size_prior) + pt.log1p(-lambda_0)
    log_sizes = log_size_prior[None, :] + size_log_evidence  # (sets, widths)
    log_rules = log_rule_prior[None, :] + rule_log_lik  # (sets, rules)
    log_total = pt.logaddexp(pt.logsumexp(log_rules, axis=1), pt.logsumexp(log_sizes, axis=1))
    rule_post = pt.exp(log_rules - log_total[:, None])
    size_post = pt.exp(log_sizes - log_total[:, None])
    in_concept = (rule_post[cell_set] * cell_rule_in).sum(axis=1) + (size_post[cell_set] * cell_size_in).sum(axis=1)

    p_concept = pt.clip(in_concept[row_cell], 0.0, 1.0)
    lapse = pm.Beta("lapse", 2, 8)  # share of trials answered by guessing
    guess_yes = pm.Beta("guess_yes", 2, 2)  # P(yes) when guessing
    p_yes = pm.Deterministic("p_yes", pt.clip((1 - lapse) * p_concept + lapse * guess_yes, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    all_log_lik = np.stack([set_log_lik(s) for s in unique])
    rule_ll = all_log_lik[:, :N_RULES]
    interval_ll = all_log_lik[:, N_RULES:]
    size_log_evidence = np.zeros((len(unique), 100))
    size_in_concept = np.zeros((len(unique), 100, 100))
    for k, s in enumerate(SIZES):
        cols = np.where(INTERVAL_SIZE == s)[0]
        ll = interval_ll[:, cols]
        lse = logsumexp(ll, axis=1)
        size_log_evidence[:, k] = lse - np.log(len(cols))
        size_in_concept[:, k, :] = np.exp(ll - lse[:, None]) @ MEMBERSHIP[N_RULES + cols]
    cells = [(index[st], int(r["target"]) - 1) for st, r in zip(sets, rows)]
    unique_cells = list(dict.fromkeys(cells))
    cell_index = {c: i for i, c in enumerate(unique_cells)}
    cs = np.array([c[0] for c in unique_cells], dtype="int64")
    ct = np.array([c[1] for c in unique_cells], dtype="int64")
    return {
        "rule_log_lik": rule_ll,
        "size_log_evidence": size_log_evidence,
        "cell_set": cs,
        "cell_rule_in": MEMBERSHIP[:N_RULES][:, ct].T.copy(),
        "cell_size_in": size_in_concept[cs, :, ct],
        "row_cell": np.array([cell_index[c] for c in cells], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
