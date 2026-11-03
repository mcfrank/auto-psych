"""Set-size-scaled evidence on the heading / Weber / upward-counting with-replacement learner.

Refinement of `heading_weber_upward_with_replacement`: people do Bayesian concept
learning over 45 arithmetic and digit rules, the 18 digit-sum concepts (an inferred
family share) and every interval of 1..100. They read the examples as independent
draws with 10% strays, guess on some trials, use a Weber range-width prior
(width / midpoint)^gamma, stretch ranges toward the last step's heading and credit
them by their room above the largest example. The one change is that the weight of
each example's evidence depends on how many examples were shown: the log-likelihood
of an n-example set is scaled by (n / 2.5)^(-rho), with rho inferred. So a lone
example is a weaker (or stronger) hint than examples that come in a larger,
agreeing set. Everything else is as in the incumbent.
"""

import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

ALPHA = 0.9
NUMBERS = np.arange(1, 101)
K_G = 7  # Taylor terms in gamma * scaled log relative width
K_R = 6  # Taylor terms in each of the room-above and room-below exponents
K_E = 7  # Taylor terms in the evidence-weight exponent
E_SCALE = 10.0  # log-likelihoods relative to the set's best range are divided by this


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
FACT_E = np.array([math.factorial(k) for k in range(K_E)], dtype="float64")
N_MOM = K_G * K_R * K_R


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


# Scaling the evidence by w multiplies each range's weight exp(ll - scale) by
# exp((w - 1) * (ll - scale)), a fourth Taylor series (in (w - 1) * E_SCALE) on top of
# the incumbent's (relative width, above, below) series; prepare_observed tabulates the
# moments with this extra axis, and the model contracts it with one coefficient
# vector per set size.

with pm.Model() as model:
    finite_log_lik = pm.Data("finite_log_lik", np.zeros((1, N_FINITE)))  # (sets, rules + sums)
    interval_log_scale = pm.Data("interval_log_scale", np.zeros(1))  # (sets,)
    prior_moments = pm.Data("prior_moments", np.zeros((1, N_MOM)))  # (sets, moments)
    evidence_moments = pm.Data("evidence_moments", np.zeros((1, K_E, N_MOM)))  # (sets, K_E, moments)
    member_moments = pm.Data("member_moments", np.zeros((1, K_E, N_MOM)))  # (cells, K_E, moments)
    set_heading = pm.Data("set_heading", np.zeros(1, dtype="int64"))  # (sets,) 0: down, 1: none, 2: up
    set_size = pm.Data("set_size", np.ones(1, dtype="int64"))  # (sets,) number of examples
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # (cells,)
    cell_target = pm.Data("cell_target", np.zeros(1, dtype="int64"))  # (cells,) target - 1
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # (trials,)
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    gamma = pm.Normal("gamma", 0.0, 0.5)  # bias on relative width of the range prior
    beta = pm.Normal("beta", 0.0, 1.0)  # pull of the ranges toward the heading direction
    delta = pm.Normal("delta", 0.0, 0.5)  # order-independent credit for room above the examples
    log_sum_share = pm.Normal("log_sum_share", -3.0, 1.5)
    rho = pm.Normal("rho", 0.0, 0.3)  # evidence weight per example scales as (n / 2.5)^(-rho)

    size_weight = pt.exp(-rho * np.log(np.arange(1, 5) / 2.5))  # (4,) weight for 1..4 examples
    coef_e = pt.stack([taylor_coef((size_weight[n] - 1.0) * E_SCALE, K_E, FACT_E) for n in range(4)])
    coef_g = taylor_coef(gamma * REL_HALF, K_G, FACT_G)
    coefs = []
    for hd in (-1.0, 0.0, 1.0):
        coef_a = taylor_coef(delta + hd * beta, K_R, FACT_R)
        coef_b = taylor_coef(-hd * beta, K_R, FACT_R) if hd != 0.0 else pt.as_tensor(np.eye(K_R)[0])
        coefs.append((coef_g[:, None, None] * coef_a[None, :, None] * coef_b[None, None, :]).reshape((-1,)))
    coef = pt.stack(coefs, axis=1)  # (moments, 3 heading classes)
    n_sets = set_heading.shape[0]
    n_cells = cell_set.shape[0]
    set_rows = pt.arange(n_sets)
    cell_rows = pt.arange(n_cells)
    weight = size_weight[set_size - 1]  # (sets,)

    log_z = pt.log(pt.dot(prior_moments, coef)[set_rows, set_heading])
    evid_by_e = pt.dot(evidence_moments.reshape((-1, N_MOM)), coef).reshape((n_sets, K_E, 3))
    evid_by_e = evid_by_e[set_rows, :, set_heading]  # (sets, K_E)
    evidence = pt.sum(evid_by_e * coef_e[set_size - 1], axis=1)
    interval_log_evidence = pt.log(evidence) + weight * interval_log_scale - log_z
    cell_heading = set_heading[cell_set]
    member_by_e = pt.dot(member_moments.reshape((-1, N_MOM)), coef).reshape((n_cells, K_E, 3))
    member_by_e = member_by_e[cell_rows, :, cell_heading]  # (cells, K_E)
    member_mass = pt.sum(member_by_e * coef_e[set_size[cell_set] - 1], axis=1)
    interval_in_cell = member_mass / evidence[cell_set]

    scaled = weight[:, None] * finite_log_lik
    rule_ll, sum_ll = scaled[:, :N_RULES], scaled[:, N_RULES:]
    rule_ev = pt.logsumexp(rule_ll, axis=1)
    sum_ev = pt.logsumexp(sum_ll, axis=1)
    rule_in_set = pt.dot(pt.special.softmax(rule_ll, axis=1), MEMBERSHIP[:N_RULES])  # (sets, 100)
    sum_in_set = pt.dot(pt.special.softmax(sum_ll, axis=1), MEMBERSHIP[N_RULES:N_FINITE])
    rule_in_cell = rule_in_set[cell_set, cell_target]
    sum_in_cell = sum_in_set[cell_set, cell_target]

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
    """sum_i rel_w[i,k] above_p[i,a] below_p[i,b] -> flat (K_G * K_R * K_R,)."""
    ra = (rel_w[:, :, None] * above_p[:, None, :]).reshape(len(rel_w), -1)
    return (ra.T @ below_p).reshape(-1)


def evidence_joint_moments(weighted, e_powers, above_p, below_p):
    """Moments with the evidence-exponent axis in front -> (K_E, K_G * K_R * K_R)."""
    return np.stack([joint_moments(weighted * e_powers[:, k : k + 1], above_p, below_p) for k in range(K_E)])


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    set_index = {s: i for i, s in enumerate(unique)}
    keys = [(set_index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: i for i, c in enumerate(cells)}
    n_sets, n_cells = len(unique), len(cells)
    finite_ll = np.zeros((n_sets, N_FINITE))
    scale = np.zeros(n_sets)
    heading = np.zeros(n_sets, dtype="int64")
    prior_m = np.zeros((n_sets, N_MOM))
    evid_m = np.zeros((n_sets, K_E, N_MOM))
    member_m = np.zeros((n_cells, K_E, N_MOM))
    for i, s in enumerate(unique):
        ll = set_log_lik(s)
        finite_ll[i] = ll[:N_FINITE]
        int_ll = ll[N_FINITE:]
        scale[i] = int_ll.max()
        heading[i] = int(heading_of(s)) + 1
        above, below = rooms(s)
        above_p = above[:, None] ** np.arange(K_R)[None, :]
        below_p = below[:, None] ** np.arange(K_R)[None, :]
        e_powers = ((int_ll - scale[i]) / E_SCALE)[:, None] ** np.arange(K_E)[None, :]
        prior_m[i] = joint_moments(REL_POWERS, above_p, below_p)
        weighted = np.exp(int_ll - scale[i])[:, None] * REL_POWERS
        evid_m[i] = evidence_joint_moments(weighted, e_powers, above_p, below_p)
        for t in [t for (si, t) in cells if si == i]:
            inside = INTERVAL_MEMBERS[:, t - 1] > 0
            member_m[cell_index[(i, t)]] = evidence_joint_moments(
                weighted[inside], e_powers[inside], above_p[inside], below_p[inside]
            )
    cell_set = np.array([c[0] for c in cells], dtype="int64")
    return {
        "finite_log_lik": finite_ll,
        "interval_log_scale": scale,
        "prior_moments": prior_m,
        "evidence_moments": evid_m,
        "member_moments": member_m,
        "set_heading": heading,
        "set_size": np.array([len(s) for s in unique], dtype="int64"),
        "cell_set": cell_set,
        "cell_target": np.array([c[1] - 1 for c in cells], dtype="int64"),
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


SAMPLER_SETTINGS = {"target_accept": 0.9}
