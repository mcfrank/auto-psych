"""Heading, Weber width and upward counting with evidence weighted by set size.

Refinement of `heading_weber_upward_with_replacement`: people do Bayesian
concept learning over 45 arithmetic and digit rules, the 18 digit-sum concepts
(an inferred family share) and every interval of 1..100, reading the examples as
independent draws (with replacement) with 10% strays and a guessing lapse, with
a Weber range-width prior (width / midpoint)^gamma, ranges stretched toward the
last step's heading and credited by their room above the largest example. The
one change is that the weight people give the evidence depends on how many
examples were shown: the log-likelihood of an n-example set enters belief scaled
by n^(-kappa) (kappa inferred), so single examples are judged as before, while
larger sets sharpen belief toward small concepts by more (kappa < 0) or by less
(kappa > 0) than an ideal learner.

The interval part keeps the incumbent's Taylor-moment tabulation, with one more
expansion axis: exp(t L) = exp(t L_max) exp(L') exp((t - 1) L'), where
L' = L - L_max, expanded in powers of (t - 1).
"""

import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

ALPHA = 0.9
NUMBERS = np.arange(1, 101)
K_G = 8  # Taylor terms in gamma * scaled log relative width
K_R = 6  # Taylor terms in each of the room-above and room-below exponents
K_T = 5  # Taylor terms in (evidence weight - 1) * log-likelihood
L_FLOOR = -20.0  # intervals this far below the best one carry no weight


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
# log(width / midpoint), centred and scaled to [-1, 1]
LOG_REL = np.log((INTERVAL_HI - INTERVAL_LO + 1) / ((INTERVAL_LO + INTERVAL_HI) / 2))
REL_CENTRE = (LOG_REL.max() + LOG_REL.min()) / 2
REL_HALF = (LOG_REL.max() - LOG_REL.min()) / 2
REL_SCALED = (LOG_REL - REL_CENTRE) / REL_HALF
REL_POWERS = REL_SCALED[:, None] ** np.arange(K_G)[None, :]  # (intervals, K_G)
FACT_G = np.array([math.factorial(k) for k in range(K_G)], dtype="float64")
FACT_R = np.array([math.factorial(k) for k in range(K_R)], dtype="float64")
FACT_T = np.array([math.factorial(k) for k in range(K_T)], dtype="float64")
N_PRIOR_MOM = K_G * K_R * K_R
N_EVID_MOM = N_PRIOR_MOM * K_T


def set_log_lik(examples):
    """log P(examples in order | h) for every hypothesis, sampling members with replacement."""
    total = np.zeros(len(MEMBERSHIP))
    for x in examples:
        total += np.log(ALPHA * MEMBERSHIP[:, x - 1] / SIZE + (1 - ALPHA) / 100)
    return total


def heading_of(examples):
    return float(np.sign(examples[-1] - examples[-2])) if len(examples) > 1 else 0.0


def rooms(examples):
    """Room above the largest and below the smallest example for every interval, /100."""
    above = np.clip(INTERVAL_HI - max(examples), 0, None) / 100.0
    below = np.clip(min(examples) - INTERVAL_LO, 0, None) / 100.0
    return above, below


def taylor_coef(x, n_terms, factorial):
    """x^k / k! for k < n_terms (k = 0 kept apart: d(x^0)/dx is NaN at x = 0)."""
    return pt.concatenate([pt.ones(1), x ** np.arange(1, n_terms)]) / factorial


# Classes: heading (down, none, up) x set size (1..4), index 4 * heading + (n - 1).
with pm.Model() as model:
    rule_ll = pm.Data("rule_ll", np.zeros((1, N_RULES)))  # (sets, rules)
    sum_ll = pm.Data("sum_ll", np.zeros((1, N_SUMS)))  # (sets, digit sums)
    rule_mem = pm.Data("rule_mem", np.zeros((1, N_RULES)))  # (cells, rules)
    sum_mem = pm.Data("sum_mem", np.zeros((1, N_SUMS)))  # (cells, digit sums)
    interval_log_scale = pm.Data("interval_log_scale", np.zeros(1))  # (sets,)
    prior_moments = pm.Data("prior_moments", np.zeros((1, N_PRIOR_MOM)))  # (sets, moments)
    evidence_moments = pm.Data("evidence_moments", np.zeros((1, N_EVID_MOM)))  # (sets, moments)
    member_moments = pm.Data("member_moments", np.zeros((1, N_EVID_MOM)))  # (cells, moments)
    set_heading = pm.Data("set_heading", np.zeros(1, dtype="int64"))  # (sets,) 0: down, 1: none, 2: up
    set_size = pm.Data("set_size", np.ones(1, dtype="int64"))  # (sets,) 1..4
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # (cells,)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # (trials,)
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    gamma = pm.Normal("gamma", 0.0, 0.5)  # bias on relative width of the range prior
    beta = pm.Normal("beta", 0.0, 1.0)  # pull of the ranges toward the heading direction
    delta = pm.Normal("delta", 0.0, 0.5)  # order-independent credit for room above the examples
    log_sum_share = pm.Normal("log_sum_share", -3.0, 1.5)
    kappa = pm.Normal("kappa", 0.0, 0.2)  # evidence weight per set is n^(-kappa)

    weight_by_n = pt.exp(-kappa * np.log(np.arange(1.0, 5.0)))  # (4,)
    coef_g = taylor_coef(gamma * REL_HALF, K_G, FACT_G)
    prior_coefs, evid_coefs = [], []
    for hd in (-1.0, 0.0, 1.0):
        coef_a = taylor_coef(delta + hd * beta, K_R, FACT_R)
        coef_b = taylor_coef(-hd * beta, K_R, FACT_R) if hd != 0.0 else pt.as_tensor(np.eye(K_R)[0])
        coef3 = (coef_g[:, None, None] * coef_a[None, :, None] * coef_b[None, None, :]).reshape((-1,))
        prior_coefs.append(coef3)
        for n in range(4):
            coef_t = taylor_coef(weight_by_n[n] - 1.0, K_T, FACT_T) if n > 0 else pt.as_tensor(np.eye(K_T)[0])
            evid_coefs.append((coef3[:, None] * coef_t[None, :]).reshape((-1,)))
    prior_coef = pt.stack(prior_coefs, axis=0)  # (3, prior moments)
    evid_coef = pt.stack(evid_coefs, axis=0)  # (12, evidence moments)

    set_class = 4 * set_heading + set_size - 1
    t_set = weight_by_n[set_size - 1]
    set_rows = pt.arange(set_class.shape[0])
    cell_rows = pt.arange(cell_set.shape[0])
    log_z = pt.log(pt.dot(prior_moments, prior_coef.T)[set_rows, set_heading])
    evidence = pt.dot(evidence_moments, evid_coef.T)[set_rows, set_class]
    interval_log_evidence = pt.log(evidence) + t_set * interval_log_scale - log_z
    member_mass = pt.dot(member_moments, evid_coef.T)[cell_rows, set_class[cell_set]]
    interval_in_cell = member_mass / evidence[cell_set]

    rule_w = t_set[:, None] * rule_ll
    sum_w = t_set[:, None] * sum_ll
    rule_ev = pt.logsumexp(rule_w, axis=1)
    sum_ev = pt.logsumexp(sum_w, axis=1)
    rule_in_cell = pt.sum(pt.exp(rule_w - rule_ev[:, None])[cell_set] * rule_mem, axis=1)
    sum_in_cell = pt.sum(pt.exp(sum_w - sum_ev[:, None])[cell_set] * sum_mem, axis=1)

    log_rules_only = rule_ev - np.log(N_RULES)
    log_sums_only = sum_ev - np.log(N_SUMS) + log_sum_share
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


def joint_moments(rel_w, above_p, below_p):
    """sum_i rel_w[i,k] above_p[i,a] below_p[i,b] -> flat (len(k) * K_R * K_R,)."""
    ra = (rel_w[:, :, None] * above_p[:, None, :]).reshape(len(rel_w), -1)
    return (ra.T @ below_p).reshape(-1)


def evidence_joint(weighted_rel, l_shift, above_p, below_p):
    """Moments over (rel width, above, below, log-lik power), flat in the model's order."""
    lpow = l_shift[:, None] ** np.arange(K_T)[None, :]
    out = np.stack([joint_moments(weighted_rel * lpow[:, j : j + 1], above_p, below_p) for j in range(K_T)])
    return out.T.reshape(-1)  # (prior moment, power) -> prior-major, power-minor


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    set_index = {s: i for i, s in enumerate(unique)}
    keys = [(set_index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: i for i, c in enumerate(cells)}
    n_sets, n_cells = len(unique), len(cells)
    rule_ll = np.zeros((n_sets, N_RULES))
    sum_ll = np.zeros((n_sets, N_SUMS))
    scale = np.zeros(n_sets)
    heading = np.zeros(n_sets, dtype="int64")
    size = np.zeros(n_sets, dtype="int64")
    prior_m = np.zeros((n_sets, N_PRIOR_MOM))
    evid_m = np.zeros((n_sets, N_EVID_MOM))
    member_m = np.zeros((n_cells, N_EVID_MOM))
    for i, s in enumerate(unique):
        ll = set_log_lik(s)
        rule_ll[i], sum_ll[i], int_ll = ll[:N_RULES], ll[N_RULES:N_FINITE], ll[N_FINITE:]
        scale[i] = int_ll.max()
        heading[i] = int(heading_of(s)) + 1
        size[i] = len(s)
        above, below = rooms(s)
        above_p = above[:, None] ** np.arange(K_R)[None, :]
        below_p = below[:, None] ** np.arange(K_R)[None, :]
        prior_m[i] = joint_moments(REL_POWERS, above_p, below_p)
        l_shift = int_ll - scale[i]
        keep = l_shift > L_FLOOR
        weighted = np.exp(l_shift)[:, None] * REL_POWERS
        evid_m[i] = evidence_joint(weighted[keep], l_shift[keep], above_p[keep], below_p[keep])
        for t in [t for (si, t) in cells if si == i]:
            inside = keep & (INTERVAL_MEMBERS[:, t - 1] > 0)
            member_m[cell_index[(i, t)]] = evidence_joint(
                weighted[inside], l_shift[inside], above_p[inside], below_p[inside]
            )
    cell_set = np.array([c[0] for c in cells], dtype="int64")
    cell_target = np.array([c[1] - 1 for c in cells], dtype="int64")
    return {
        "rule_ll": rule_ll,
        "sum_ll": sum_ll,
        "rule_mem": MEMBERSHIP[:N_RULES][:, cell_target].T.copy(),
        "sum_mem": MEMBERSHIP[N_RULES:N_FINITE][:, cell_target].T.copy(),
        "interval_log_scale": scale,
        "prior_moments": prior_m,
        "evidence_moments": evid_m,
        "member_moments": member_m,
        "set_heading": heading,
        "set_size": size,
        "cell_set": cell_set,
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


SAMPLER_SETTINGS = {"target_accept": 0.9}
