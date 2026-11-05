"""Outlier-tolerant concept learning (refines bigelow_piantadosi_independent).

People do normative Bayesian concept learning over 45 rules (each with its own
prior weight) and all intervals of 1-100 with the size principle, but they
mis-estimate how the examples were generated: they believe a larger share of
examples are random strays drawn uniformly from 1-100 than the ideal learner's
fixed 10%. The single distortion is this inferred noise belief (1 - alpha).
A stray-tolerant learner keeps believing in tight concepts (a small rule or a
narrow interval) despite an example that does not fit, so belief does not
spread to broad hypotheses. "Yes" is the posterior probability that the
concept contains the target.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)


def _is_prime(n):
    return n > 1 and all(n % a for a in range(2, int(n ** 0.5) + 1))


def rule_concepts():
    rules = {
        "odd": [n for n in NUMBERS if n % 2 == 1],
        "primes": [n for n in NUMBERS if _is_prime(n)],
        "squares": [n * n for n in range(1, 11)],
        "cubes": [n ** 3 for n in range(1, 5)],
        "powers of 2, not 32": [2, 4, 8, 16, 64],
        "powers of 2, or 37": [2, 4, 8, 16, 32, 37, 64],
        "even": list(range(2, 101, 2)),
    }
    rules.update({f"multiples of {m}": list(range(m, 101, m)) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]})
    rules.update({f"powers of {m}": [m ** k for k in range(1, 8) if m ** k <= 100] for m in range(2, 11)})
    rules.update({f"ends in {d}": [n for n in NUMBERS if n % 10 == d] for d in range(10)})
    rules.update({f"contains digit {d}": [n for n in NUMBERS if str(d) in str(n)] for d in range(10)})
    return rules


_RULES = rule_concepts()
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
SIZES = MEMBERSHIP.sum(axis=1)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]

# "ends in 0" and "contains digit 0" are the same set: sample their summed
# weight plus a Beta split (Dirichlet aggregation keeps Dirichlet(2,...,2)).
END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

with pm.Model() as model:
    n_in = pm.Data("n_in", np.zeros((1, len(SIZES))))  # (sets, hypotheses): examples inside h
    n_out = pm.Data("n_out", np.zeros((1, len(SIZES))))  # examples outside h
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    set_weights = pm.Dirichlet("set_weights", a=SET_CONCENTRATION)
    zero_split = pm.Beta("zero_split", 2, 2)
    shared = set_weights[END_0]
    rule_weights = pm.Deterministic("rule_weights", pt.concatenate([
        set_weights[:END_0], (zero_split * shared)[None], set_weights[END_0 + 1:CONTAINS_0],
        ((1 - zero_split) * shared)[None], set_weights[CONTAINS_0:],
    ]))
    # Believed probability that an example was sampled from the concept
    # (the ideal learner uses 0.9); 1 - alpha is the believed stray rate.
    alpha = pm.Beta("alpha", 3.0, 1.5)

    log_in = pt.log(alpha / pt.as_tensor(SIZES)[None, :] + (1 - alpha) / 100.0)
    log_out = pt.log((1 - alpha) / 100.0)
    log_lik = n_in * log_in + n_out * log_out  # (sets, hypotheses)
    rule_log_lik = log_lik[:, :N_RULES]
    interval_log_lik = log_lik[:, N_RULES:]

    int_log_evidence = pt.logsumexp(interval_log_lik, axis=1)
    int_post = pt.exp(interval_log_lik - int_log_evidence[:, None])
    interval_in_concept = pt.dot(int_post, pt.as_tensor(INTERVAL_MEMBERSHIP))

    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + rule_log_lik
    log_intervals = pt.log1p(-lambda_0) - 2 * np.log(100) + int_log_evidence
    log_joint = pt.concatenate([log_rules, log_intervals[:, None]], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = (posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP)
                  + posterior[:, N_RULES][:, None] * interval_in_concept)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    n_in = np.stack([MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique])
    n_out = np.array([len(s) for s in unique], dtype="float64")[:, None] - n_in
    return {
        "n_in": n_in,
        "n_out": n_out,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
