"""Justification search (noisy-OR retrieval of explanations).

People do not weigh competing explanations against one another; they answer by
searching memory for any familiar explanation (a rule such as "even" or "ends
in 6", or a numerical range) that covers both the examples and the target, and
say "yes" as soon as one comes to mind. Each explanation comes to mind with a
chance that grows with how well it accounts for the examples on its own (its
prior plausibility times how much likelier the examples are under it than as
random numbers from 1-100), so the target is endorsed by the absolute evidence
for explanations that include it — a Poisson noisy-OR over retrieved
explanations — not by its share relative to better explanations that exclude
it (no normalisation across concepts, unlike Bayesian averaging). A fitted
retrieval rate scales the search; a fitted base rate and ceiling capture
answers given without any retrieved explanation and slips on certain members.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_N = 4
G_MAX = 480


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


RULE_NAMES = list(rule_concepts().keys())
RULES = list(rule_concepts().values())
IS_DIGIT_RULE = np.array([n.startswith("ends in") or n.startswith("contains digit") for n in RULE_NAMES], dtype="float64")
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
LO = np.array([m for m in NUMBERS for n in range(m, 101)])
HI = np.array([n for m in NUMBERS for n in range(m, 101)])
INTERVAL_SIZE = (HI - LO + 1).astype("float64")
WIDTHS = np.arange(1, 101, dtype="float64")
LOG_WIDTH_COUNT = np.log(101 - WIDTHS)
N_HYP = N_RULES + G_MAX

with pm.Model() as model:
    contains = pm.Data("contains", np.ones((1, MAX_N, N_HYP)))  # example k lies in hypothesis
    step_mask = pm.Data("step_mask", np.zeros((1, MAX_N)))  # example k exists
    group_size = pm.Data("group_size", np.ones((1, G_MAX)))  # width of the group's intervals
    pair_set = pm.Data("pair_set", np.zeros(1, dtype="int64"))
    pair_rule_log_in = pm.Data("pair_rule_log_in", np.zeros((1, N_RULES)))  # 0 if rule has target, else -1e3
    pair_group_log_n = pm.Data("pair_group_log_n", np.zeros((1, G_MAX)))  # log #intervals in group with target
    row_pair = pm.Data("row_pair", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    log_sigma = pm.Normal("log_sigma", mu=np.log(10.0), sigma=1.0)
    sigma = pt.exp(log_sigma)
    log_digit_w = pm.Normal("log_digit_w", mu=0.0, sigma=1.5)
    rule_lw = log_digit_w * pt.as_tensor(IS_DIGIT_RULE)
    rule_log_prior = pt.log(lambda_0) + rule_lw - pt.logsumexp(rule_lw)  # (rules,)
    log_norm = pt.logsumexp(LOG_WIDTH_COUNT + np.log(WIDTHS) - WIDTHS / sigma)
    interval_log_prior = pt.log1p(-lambda_0) + pt.log(group_size) - group_size / sigma - log_norm  # per interval

    # Evidence each explanation carries on its own: examples' likelihood under
    # it relative to random numbers from 1-100.
    alpha = pm.Beta("alpha", 9, 1)
    n_sets = group_size.shape[0]
    hyp_size = pt.concatenate([pt.zeros((n_sets, N_RULES)) + pt.as_tensor(RULE_SIZE)[None, :], group_size], axis=1)
    log_in = pt.log(alpha * 100.0 / hyp_size + (1 - alpha))[:, None, :]
    log_out = pt.log1p(-alpha)
    log_lr = (step_mask[:, :, None] * (contains * log_in + (1 - contains) * log_out)).sum(axis=1)  # (sets, hyp)

    rule_lw_set = rule_log_prior[None, :] + log_lr[:, :N_RULES]
    group_lw_set = interval_log_prior + log_lr[:, N_RULES:]
    log_mass = pt.logsumexp(
        pt.concatenate([pt.take(rule_lw_set, pair_set, axis=0) + pair_rule_log_in,
                        pt.take(group_lw_set, pair_set, axis=0) + pair_group_log_n], axis=1),
        axis=1,
    )  # log evidence for explanations covering the target

    # Retrieval: Poisson noisy-OR over explanations covering examples and target.
    log_rate = pm.Normal("log_rate", mu=0.0, sigma=3.0)
    p_found = 1 - pt.exp(-pt.exp(pt.clip(log_rate + log_mass, -30.0, 30.0)))
    base = pm.Beta("base", 1, 9)
    ceiling = pm.Beta("ceiling", 8, 2)
    p_pair = ceiling * (base + (1 - base) * p_found)
    p_yes = pm.Deterministic("p_yes", pt.clip(p_pair[row_pair], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    n_sets = len(unique)
    contains = np.ones((n_sets, MAX_N, N_HYP))
    mask = np.zeros((n_sets, MAX_N))
    gsize = np.ones((n_sets, G_MAX))
    groups = []
    for i, s in enumerate(unique):
        inside = np.stack([(LO <= x) & (HI >= x) for x in s], axis=1)
        code = (inside * (1 << np.arange(len(s)))).sum(axis=1)
        keys, group = np.unique(np.stack([INTERVAL_SIZE.astype(int), code], axis=1), axis=0, return_inverse=True)
        group = group.reshape(-1)
        n_g = len(keys)
        assert n_g <= G_MAX
        groups.append(group)
        gsize[i, :n_g] = keys[:, 0]
        for k in range(len(s)):
            contains[i, k, :N_RULES] = RULE_MEMBERSHIP[:, s[k] - 1]
            contains[i, k, N_RULES:] = 0.0
            contains[i, k, N_RULES:N_RULES + n_g] = (keys[:, 1] >> k) & 1
            mask[i, k] = 1.0
    pairs = list(dict.fromkeys(zip([index[s] for s in sets], targets)))
    pair_index = {p: j for j, p in enumerate(pairs)}
    rule_in = np.zeros((len(pairs), N_RULES))
    group_n = np.zeros((len(pairs), G_MAX))
    for j, (i, t) in enumerate(pairs):
        rule_in[j] = np.where(RULE_MEMBERSHIP[:, t - 1] > 0, 0.0, -1e3)
        hit = (LO <= t) & (HI >= t)
        cnt = np.bincount(groups[i][hit], minlength=G_MAX)[:G_MAX].astype(float)
        group_n[j] = np.where(cnt > 0, np.log(np.maximum(cnt, 1.0)), -1e3)
    return {
        "contains": contains,
        "step_mask": mask,
        "group_size": gsize,
        "pair_set": np.array([p[0] for p in pairs], dtype="int64"),
        "pair_rule_log_in": rule_in,
        "pair_group_log_n": group_n,
        "row_pair": np.array([pair_index[(index[s], t)] for s, t in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
