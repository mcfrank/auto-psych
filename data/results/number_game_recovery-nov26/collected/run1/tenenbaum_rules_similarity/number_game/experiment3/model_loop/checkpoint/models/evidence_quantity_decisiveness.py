"""Evidence-quantity confidence: decisiveness grows with the number of examples.

People infer the hidden program as a Bayesian concept learner over familiar
number rules (all equally plausible a priori, one shared mass), digit-sum rules
(prior share proportional to exp(-decay * s)) and the 5,050 ranges of 1..100
(Erlang width prior, the size principle, 10% of examples uniform noise). But
how decisively they act on that belief depends on how much evidence they saw:
their confidence scales with the number of examples, so the log-odds of the
belief are multiplied by a gain exp(g * (n - 1)) before answering: a single
example yields probability matching, and each further example makes answers
more (g > 0) or less (g < 0) decisive. With g > 0, after four examples a
partly-believed target is pushed toward a firm "yes" or "no", while after two
the same belief gets a more hedged answer; g = 0 recovers the ideal learner.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
NUMBERS = np.arange(1, 101)


def familiar_rules():
    rules = {
        "odd": [n for n in NUMBERS if n % 2 == 1],
        "even": list(range(2, 101, 2)),
        "primes": [n for n in NUMBERS if n > 1 and all(n % a for a in range(2, int(n**0.5) + 1))],
        "squares": [n * n for n in range(1, 11)],
        "cubes": [n**3 for n in range(1, 5)],
    }
    rules.update({f"multiples of {m}": list(range(m, 101, m)) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]})
    rules.update({f"powers of {m}": [m**k for k in range(1, 8) if m**k <= 100] for m in range(2, 11)})
    rules.update({f"ends in {d}": [n for n in NUMBERS if n % 10 == d] for d in range(10)})
    return list(rules.values())


DSUM_RULES = [[n for n in NUMBERS if sum(map(int, str(n))) == t] for t in range(1, 19)]
RULES = familiar_rules() + DSUM_RULES
N_FAMILIAR = len(RULES) - 18
N_RULES = len(RULES)
DSUM_VALUE = np.arange(1, 19, dtype="float64")
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_LOG_LIK = np.log(ALPHA * RULE_MEMBERSHIP / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
LO, HI = np.array([(m, n) for m in NUMBERS for n in range(m, 101)]).T
WIDTH = HI - LO + 1
WIDTHS = np.arange(1, 101, dtype="float64")
N_PER_WIDTH = 101.0 - WIDTHS
LOG_IN = np.log(ALPHA / WIDTHS + (1 - ALPHA) / 100)
LOG_OUT = np.log((1 - ALPHA) / 100)
GROUP = (WIDTH - 1).astype("int64")  # ranges of one width share a prior

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    range_log_total = pm.Data("range_log_total", np.zeros((1, 100)))  # (sets, widths)
    range_in_share = pm.Data("range_in_share", np.zeros((1, 100, 100)))  # (sets, widths, targets)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    set_size = pm.Data("set_size", np.zeros(1, dtype="float64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Prior plausibility: kind shares (softmax logits), Erlang widths, graded sums.
    kind_logit = pm.Normal("kind_logit", mu=0.0, sigma=1.5, shape=2)
    _l = pt.concatenate([kind_logit, pt.zeros(1)])
    log_w = _l - pt.logsumexp(_l)
    interval_scale = pm.LogNormal("interval_scale", mu=np.log(20.0), sigma=1.0)
    dsum_decay = pm.Normal("dsum_decay", mu=0.0, sigma=0.5)
    # Change in decisiveness (log gain on the belief's log-odds) per extra example.
    gain_per_example = pm.Normal("gain_per_example", mu=0.0, sigma=0.5)

    erlang = np.log(WIDTHS) - WIDTHS / interval_scale
    log_prior_range = log_w[2] + erlang - pt.logsumexp(erlang + np.log(N_PER_WIDTH))  # per range of width W
    dsum_log_share = -dsum_decay * DSUM_VALUE
    dsum_log_share = dsum_log_share - pt.logsumexp(dsum_log_share)
    log_prior_rules = pt.concatenate(
        [pt.zeros(N_FAMILIAR) + log_w[0] - np.log(N_FAMILIAR), log_w[1] + dsum_log_share]
    )

    log_rules = log_prior_rules[None, :] + rule_log_lik
    log_ranges = log_prior_range[None, :] + range_log_total
    log_joint = pt.concatenate([log_rules, log_ranges], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP) + (
        posterior[:, N_RULES:, None] * range_in_share
    ).sum(axis=1)  # (sets, targets)

    belief = pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6)
    gain = pt.exp(gain_per_example * (set_size - 1.0))
    p_yes = pm.Deterministic("p_yes", pt.clip(pm.math.sigmoid(gain * (pt.log(belief) - pt.log1p(-belief))), 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = np.array([int(r["target"]) for r in rows])
    unique = list(dict.fromkeys(sets))
    lookup = {s: i for i, s in enumerate(unique)}
    index = np.array([lookup[s] for s in sets])
    total = np.zeros((len(unique), 100))
    inside = np.zeros((len(unique), 100, 100))
    for i, s in enumerate(unique):
        ll = np.zeros(len(LO))
        for x in s:
            ll += np.where((LO <= x) & (x <= HI), LOG_IN[WIDTH - 1], LOG_OUT)
        top = ll.max()
        e = np.exp(ll - top)
        tot = np.bincount(GROUP, weights=e, minlength=100)
        total[i] = top + np.log(np.maximum(tot, 1e-300))
        for t in range(1, 101):
            hit = (LO <= t) & (t <= HI)
            inside[i, :, t - 1] = np.bincount(GROUP[hit], weights=e[hit], minlength=100) / np.maximum(tot, 1e-300)
    return {
        "rule_log_lik": np.stack([RULE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "range_log_total": total,
        "range_in_share": inside,
        "set_size": np.array([len(s) for s in sets], dtype="float64"),
        "row_cell": (index * 100 + targets - 1).astype("int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
