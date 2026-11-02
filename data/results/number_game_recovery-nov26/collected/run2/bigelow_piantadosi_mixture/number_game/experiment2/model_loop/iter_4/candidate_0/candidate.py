"""Uncertain strong vs. weak sampling (on the margin_beyond_examples_range_prior learner).

People are unsure how the program's examples were chosen: with inferred belief
pi they were sampled from the hidden concept (strong sampling, without
replacement: the size principle), and with belief 1 - pi they are arbitrary
numbers that merely carry the label (weak sampling: each example has
probability 1/(100 - shown) if the concept contains it, whatever the concept's
size). The likelihood of the examples under every concept is the pi-mixture of
the two, both with 10% strays. Everything else is the best model: 45 rules, 18
digit-sum concepts, every interval with width-graded, open-ended-discounted,
margin-credited prior, and a guessing lapse. Prediction: broad consistent
concepts keep a floor of credibility that does not shrink with more examples.
"""

import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

ALPHA = 0.9
NUMBERS = np.arange(1, 101)
N_TERMS = 14  # Taylor terms of exp(beta * c), 0 <= c < 1: relative error < 1e-4 for |beta| < 3


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
DIGIT_SUM = np.array([sum(int(c) for c in str(n)) for n in NUMBERS])
SUM_CONCEPTS = [list(NUMBERS[DIGIT_SUM == s]) for s in range(1, 19)]
N_SUMS = len(SUM_CONCEPTS)
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + SUM_CONCEPTS + INTERVALS], dtype="float64")
SIZE = MEMBERSHIP.sum(axis=1)
N_FINITE = N_RULES + N_SUMS
INTERVAL_MEMBERS = MEMBERSHIP[N_FINITE:]
INTERVAL_LO = np.array([iv[0] for iv in INTERVALS], dtype="float64")
INTERVAL_HI = np.array([iv[-1] for iv in INTERVALS], dtype="float64")
INTERVAL_WIDTH = (INTERVAL_HI - INTERVAL_LO + 1).astype(int)
INTERVAL_OPEN = ((INTERVAL_LO == 1) | (INTERVAL_HI == 100)).astype(int)
# Intervals are grouped by (width, open-ended flag); only non-empty groups kept.
_GROUP_KEYS = sorted(set(zip(INTERVAL_WIDTH.tolist(), INTERVAL_OPEN.tolist())))
_GROUP_OF = {g: i for i, g in enumerate(_GROUP_KEYS)}
INTERVAL_GROUP = np.array([_GROUP_OF[(w, o)] for w, o in zip(INTERVAL_WIDTH, INTERVAL_OPEN)])
N_GROUPS = len(_GROUP_KEYS)
GROUP_WIDTH = np.array([g[0] for g in _GROUP_KEYS], dtype="float64")
GROUP_OPEN = np.array([g[1] for g in _GROUP_KEYS], dtype="float64")
GROUP_ONEHOT = (INTERVAL_GROUP[:, None] == np.arange(N_GROUPS)[None, :]).astype("float64")
FACTORIAL = np.array([math.factorial(k) for k in range(N_TERMS)], dtype="float64")


def set_log_lik(examples):
    """log P(examples in order | h) for every hypothesis, sampling members without replacement."""
    total = np.zeros(len(MEMBERSHIP))
    shown = np.zeros(len(MEMBERSHIP))
    for x in examples:
        member = MEMBERSHIP[:, x - 1]
        remaining = np.maximum(SIZE - shown, 1.0)
        total += np.log(ALPHA * member / remaining + (1 - ALPHA) / 100)
        shown += member
    return total


def weak_log_lik(examples):
    """log P(examples | h) under weak sampling: distinct arbitrary numbers, labelled if in h."""
    total = np.zeros(len(MEMBERSHIP))
    for k, x in enumerate(examples):
        total += np.log(ALPHA * MEMBERSHIP[:, x - 1] / (100 - k) + (1 - ALPHA) / 100)
    return total


def margin(examples):
    """(room below the smallest example + room above the largest) / 100 for every interval."""
    above = np.clip(INTERVAL_HI - max(examples), 0, None)
    below = np.clip(min(examples) - INTERVAL_LO, 0, None)
    return (above + below) / 100.0


# Each sampling reading (strong "s", weak "w") gets its own evidence tables: the
# rule and digit-sum log evidences and posterior memberships, and the interval
# Taylor moments of the margin factor exp(beta * c) per (width, open) group. The
# model mixes the two readings per concept by the inferred belief pi.

with pm.Model() as model:
    rule_ev_s = pm.Data("rule_ev_s", np.zeros(1))  # (sets,)
    rule_ev_w = pm.Data("rule_ev_w", np.zeros(1))
    sum_ev_s = pm.Data("sum_ev_s", np.zeros(1))
    sum_ev_w = pm.Data("sum_ev_w", np.zeros(1))
    scale_s = pm.Data("scale_s", np.zeros(1))  # (sets,)
    scale_w = pm.Data("scale_w", np.zeros(1))
    prior_moments = pm.Data("prior_moments", np.zeros((1, N_GROUPS * N_TERMS)))  # (sets, groups x k)
    evid_s = pm.Data("evid_s", np.zeros((1, N_GROUPS * N_TERMS)))
    evid_w = pm.Data("evid_w", np.zeros((1, N_GROUPS * N_TERMS)))
    member_s = pm.Data("member_s", np.zeros((1, N_GROUPS * N_TERMS)))  # (cells, groups x k)
    member_w = pm.Data("member_w", np.zeros((1, N_GROUPS * N_TERMS)))
    rule_in_s = pm.Data("rule_in_s", np.zeros(1))  # (cells,)
    rule_in_w = pm.Data("rule_in_w", np.zeros(1))
    sum_in_s = pm.Data("sum_in_s", np.zeros(1))
    sum_in_w = pm.Data("sum_in_w", np.zeros(1))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # (cells,)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # (trials,)
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    gamma = pm.Normal("gamma", 0.0, 1.0)  # width bias of the range prior
    log_sum_share = pm.Normal("log_sum_share", -3.0, 1.5)
    log_open = pm.Normal("log_open", 0.0, 1.0)  # open-ended vs bounded range
    beta = pm.Normal("beta", 0.0, 1.0)  # credit for a range's reach beyond the examples
    logit_strong = pm.Normal("logit_strong", 2.0, 2.0)  # belief the examples are sampled from the concept
    log_pi = -pt.softplus(-logit_strong)
    log_1mpi = -pt.softplus(logit_strong)

    def mix(ev_s, ev_w, in_s, in_w):
        """Mixed log evidence (per set) and membership (per cell) of a concept family."""
        a, b = log_pi + ev_s, log_1mpi + ev_w
        w_s = pm.math.sigmoid(a - b)[cell_set]
        return pt.logaddexp(a, b), w_s * in_s + (1 - w_s) * in_w

    beta_powers = pt.concatenate([pt.ones(1), beta ** np.arange(1, N_TERMS)]) / FACTORIAL
    group_w = pt.exp(gamma * np.log(GROUP_WIDTH) + log_open * GROUP_OPEN)
    coef = (group_w[:, None] * beta_powers[None, :]).reshape((-1,))  # groups x k
    log_z = pt.log(pt.dot(prior_moments, coef))
    e_s, e_w = pt.dot(evid_s, coef), pt.dot(evid_w, coef)
    interval_log_evidence, interval_in_cell = mix(
        pt.log(e_s) + scale_s - log_z,
        pt.log(e_w) + scale_w - log_z,
        pt.dot(member_s, coef) / e_s[cell_set],
        pt.dot(member_w, coef) / e_w[cell_set],
    )
    rule_log_ev, rule_in_cell = mix(rule_ev_s, rule_ev_w, rule_in_s, rule_in_w)
    sum_log_ev, sum_in_cell = mix(sum_ev_s, sum_ev_w, sum_in_s, sum_in_w)

    log_rules_only = rule_log_ev - np.log(N_RULES)
    log_sums_only = sum_log_ev - np.log(N_SUMS) + log_sum_share
    p_sum = pm.math.sigmoid(log_sums_only - log_rules_only)[cell_set]
    finite_in_cell = (1 - p_sum) * rule_in_cell + p_sum * sum_in_cell
    log_finite = (
        pt.log(lambda_0) + pt.logaddexp(log_rules_only, log_sums_only) - pt.logaddexp(0.0, log_sum_share)
    )
    log_intervals = pt.log1p(-lambda_0) + interval_log_evidence
    p_rule = pm.math.sigmoid(log_finite - log_intervals)[cell_set]
    in_concept = p_rule * finite_in_cell + (1 - p_rule) * interval_in_cell

    p_concept = pt.clip(in_concept[row_cell], 0.0, 1.0)
    lapse = pm.Beta("lapse", 2, 8)
    guess_yes = pm.Beta("guess_yes", 2, 2)
    p_yes = pm.Deterministic("p_yes", pt.clip((1 - lapse) * p_concept + lapse * guess_yes, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    set_index = {s: i for i, s in enumerate(unique)}
    keys = [(set_index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: i for i, c in enumerate(cells)}
    n_sets, n_cells = len(unique), len(cells)
    cell_set = np.array([c[0] for c in cells], dtype="int64")
    cell_target = np.array([c[1] - 1 for c in cells], dtype="int64")
    out = {}
    prior_m = np.zeros((n_sets, N_GROUPS, N_TERMS))
    for i, s in enumerate(unique):
        prior_m[i] = GROUP_ONEHOT.T @ (margin(s)[:, None] ** np.arange(N_TERMS)[None, :])
    out["prior_moments"] = prior_m.reshape(n_sets, -1)
    for tag, lik in (("s", set_log_lik), ("w", weak_log_lik)):
        rule_ev, sum_ev, scale = np.zeros(n_sets), np.zeros(n_sets), np.zeros(n_sets)
        rule_in_set, sum_in_set = np.zeros((n_sets, 100)), np.zeros((n_sets, 100))
        evid_m = np.zeros((n_sets, N_GROUPS, N_TERMS))
        member_m = np.zeros((n_cells, N_GROUPS, N_TERMS))
        for i, s in enumerate(unique):
            ll = lik(s)
            rule_ll, sum_ll, int_ll = ll[:N_RULES], ll[N_RULES:N_FINITE], ll[N_FINITE:]
            rule_ev[i] = logsumexp(rule_ll)
            sum_ev[i] = logsumexp(sum_ll)
            rule_in_set[i] = np.exp(rule_ll - rule_ev[i]) @ MEMBERSHIP[:N_RULES]
            sum_in_set[i] = np.exp(sum_ll - sum_ev[i]) @ MEMBERSHIP[N_RULES:N_FINITE]
            scale[i] = int_ll.max()
            powers = margin(s)[:, None] ** np.arange(N_TERMS)[None, :]
            weighted = np.exp(int_ll - scale[i])[:, None] * powers
            evid_m[i] = GROUP_ONEHOT.T @ weighted
            for t in [t for (si, t) in cells if si == i]:
                inside = INTERVAL_MEMBERS[:, t - 1] > 0
                member_m[cell_index[(i, t)]] = GROUP_ONEHOT[inside].T @ weighted[inside]
        out.update({
            f"rule_ev_{tag}": rule_ev,
            f"sum_ev_{tag}": sum_ev,
            f"scale_{tag}": scale,
            f"evid_{tag}": evid_m.reshape(n_sets, -1),
            f"member_{tag}": member_m.reshape(n_cells, -1),
            f"rule_in_{tag}": rule_in_set[cell_set, cell_target],
            f"sum_in_{tag}": sum_in_set[cell_set, cell_target],
        })
    out["cell_set"] = cell_set
    out["row_cell"] = np.array([cell_index[k] for k in keys], dtype="int64")
    out["rating"] = np.array([int(r["rating"]) for r in rows], dtype="int64")
    return out


SAMPLER_SETTINGS = {"target_accept": 0.9}
