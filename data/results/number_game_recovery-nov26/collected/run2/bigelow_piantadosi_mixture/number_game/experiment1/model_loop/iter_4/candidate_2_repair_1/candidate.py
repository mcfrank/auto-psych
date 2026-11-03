"""People read the examples one at a time while holding a single candidate
concept (45 arithmetic and digit rules with a prior proportional to
|rule| ** gamma, a share lambda_0 for the rules, and every interval of 1..100).
Win-stay, lose-shift: the held concept is kept while each new example is one of
its members; when an example falls outside it, the learner draws a new concept
from the posterior given all examples so far (strong sampling without
replacement, 10% of examples random). "Yes" is the probability that the held
concept contains the target, with a response lapse (on a fraction `lapse` of
trials the participant guesses "yes" with probability `guess_yes`). Surviving
concepts are never re-weighed, so consistent later examples do not sharpen
belief toward narrower concepts.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

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
    return list(rules.values())


RULES = rule_concepts()
N_RULES = len(RULES)
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
CONCEPT_SIZE = MEMBERSHIP.sum(axis=1)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]
LOG_RULE_SIZE = np.log(CONCEPT_SIZE[:N_RULES])
LOG_RULE_SIZE_C = LOG_RULE_SIZE - LOG_RULE_SIZE.mean()


def prefix_log_liks(examples):
    """log P(x_1..x_k | h) for k = 1..len(examples), without replacement."""
    total = np.zeros(len(MEMBERSHIP))
    shown = np.zeros(len(MEMBERSHIP))
    seen, out = set(), []
    for x in examples:
        if x in seen:
            total = total + np.log((1 - ALPHA) / 100)
        else:
            seen.add(x)
            in_h = MEMBERSHIP[:, x - 1]
            remaining = np.maximum(CONCEPT_SIZE - shown, 1.0)
            total = total + np.log(ALPHA * in_h / remaining + (1 - ALPHA) / 100)
            shown = shown + in_h
        out.append(total.copy())
    return out


def prepare_observed(rows):
    """Per distinct set: prefix posteriors, membership of each example, and the
    fixed interval-family quantities the win-stay/lose-shift recursion needs."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    rule_ll = np.zeros((S, MAX_N, N_RULES))
    int_ev = np.zeros((S, MAX_N))
    rule_member = np.ones((S, MAX_N, N_RULES))  # padded steps: everything stays
    int_mass = np.zeros((S, MAX_N, MAX_N))  # [i, k]: mass of interval posterior_i surviving to step k
    int_target = np.zeros((S, MAX_N, 100))  # [i]: interval posterior_i surviving to the end, times membership
    for s, ex in enumerate(unique):
        lls = prefix_log_liks(ex)
        lls = lls + [lls[-1]] * (MAX_N - len(lls))
        m = [MEMBERSHIP[:, x - 1] for x in ex] + [np.ones(len(MEMBERSHIP))] * (MAX_N - len(ex))
        for k in range(MAX_N):
            rule_ll[s, k] = lls[k][:N_RULES]
            rule_member[s, k] = m[k][:N_RULES]
            ill = lls[k][N_RULES:]
            int_ev[s, k] = logsumexp(ill)
        for i in range(MAX_N):
            v = np.exp(lls[i][N_RULES:] - int_ev[s, i])
            for k in range(i, MAX_N):
                if k > i:
                    v = v * m[k][N_RULES:]
                int_mass[s, i, k] = v.sum()
            int_target[s, i] = v @ INTERVAL_MEMBERSHIP
    return {
        "rule_log_lik": rule_ll,
        "interval_log_evidence": int_ev,
        "rule_member": rule_member,
        "interval_mass": int_mass,
        "interval_target": int_target,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, MAX_N, N_RULES)))
    interval_log_evidence = pm.Data("interval_log_evidence", np.zeros((1, MAX_N)))
    rule_member = pm.Data("rule_member", np.ones((1, MAX_N, N_RULES)))
    interval_mass = pm.Data("interval_mass", np.ones((1, MAX_N, MAX_N)))
    interval_target = pm.Data("interval_target", np.zeros((1, MAX_N, 100)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    gamma = pm.Normal("gamma", 0.0, 1.0)
    log_rule_prior = gamma * LOG_RULE_SIZE_C
    log_rule_prior = log_rule_prior - pt.logsumexp(log_rule_prior)
    rule_joint = rule_log_lik + log_rule_prior[None, None, :]  # (sets, steps, rules)
    rule_log_evidence = pt.logsumexp(rule_joint, axis=2)
    rule_post = pt.exp(rule_joint - rule_log_evidence[:, :, None])
    log_rules = pt.log(lambda_0) + rule_log_evidence
    log_intervals = pt.log1p(-lambda_0) - np.log(len(INTERVALS)) + interval_log_evidence
    p_rule = pm.math.sigmoid(log_rules - log_intervals)  # (sets, steps)

    # Held-concept distribution: rules tracked explicitly, intervals by the
    # weight c_i of the draw made at step i.
    held_rules = p_rule[:, 0, None] * rule_post[:, 0]
    c = [1.0 - p_rule[:, 0]]
    for k in range(1, MAX_N):
        stay = pt.sum(held_rules * rule_member[:, k], axis=1)
        stay = stay + sum(c[i] * interval_mass[:, i, k] for i in range(k))
        shift = pt.clip(1.0 - stay, 0.0, 1.0)
        held_rules = held_rules * rule_member[:, k] + (shift * p_rule[:, k])[:, None] * rule_post[:, k]
        c.append(shift * (1.0 - p_rule[:, k]))
    in_concept = pt.dot(held_rules, RULE_MEMBERSHIP)
    in_concept = in_concept + sum(c[i][:, None] * interval_target[:, i] for i in range(MAX_N))

    p_concept = pt.clip(in_concept.reshape((-1,))[row_cell], 0.0, 1.0)
    lapse = pm.Beta("lapse", 2, 8)
    guess_yes = pm.Beta("guess_yes", 2, 2)
    p_yes = pm.Deterministic("p_yes", pt.clip((1 - lapse) * p_concept + lapse * guess_yes, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)
