"""Individual kind preferences in Bayesian concept learning.

Refinement of `equal_prior_rules_round_edge_ranges`: every participant does the
same Bayesian concept learning over equally plausible familiar number rules,
digit-sum rules (prior share changing smoothly with the sum) and numeric ranges
(Erlang width prior, round-number range edges, the size principle, 10% of
examples uniform noise). The one change is that people are not all alike in how
readily they think of the program as a magnitude range rather than a number
rule: each participant has their own prior log-odds of ranges versus rules,
drawn from a population distribution whose mean and spread are inferred (the
split between familiar rules and digit sums is shared). So some people are
"rule-seekers" who read a set as "multiples of 7" or "ends in 3", others are
"magnitude-seekers" who read the same set as a band of nearby numbers, while
which programs are considered within each kind, and how the examples weigh
them, is the same for everyone.
Participants are identified from the session order (trial_index restarting).
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
NUMBERS = np.arange(1, 101)
N_PART_MAX = 15


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


FAM_RULES = familiar_rules()
DSUM_RULES = [[n for n in NUMBERS if sum(map(int, str(n))) == t] for t in range(1, 19)]
N_FAMILIAR = len(FAM_RULES)
DSUM_VALUE = np.arange(1, 19, dtype="float64")


def membership(rules):
    return np.array([np.isin(NUMBERS, c) for c in rules], dtype="float64")


def rule_loglik(mem):
    return np.log(ALPHA * mem / mem.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)


FAM_MEM, DSUM_MEM = membership(FAM_RULES), membership(DSUM_RULES)
FAM_LL, DSUM_LL = rule_loglik(FAM_MEM), rule_loglik(DSUM_MEM)
LO, HI = np.array([(m, n) for m in NUMBERS for n in range(m, 101)]).T
WIDTH = HI - LO + 1
N_ROUND_EDGES = ((LO % 10) == 0).astype("float64") + ((HI % 10) == 0).astype("float64")
WIDTHS = np.arange(1, 101, dtype="float64")
LOG_IN = np.log(ALPHA / WIDTHS + (1 - ALPHA) / 100)
LOG_OUT = np.log((1 - ALPHA) / 100)
# Ranges sharing a width and a number of round edges share a prior.
GROUP = (3 * (WIDTH - 1) + N_ROUND_EDGES).astype("int64")
N_GROUPS = 300
GROUP_COUNT = np.bincount(GROUP, minlength=N_GROUPS).astype("float64")
GROUP_WIDTH = np.repeat(WIDTHS, 3)
GROUP_EDGES = np.tile(np.arange(3.0), 100)


def cond_kind(log_joint, pred_fn):
    """Within-kind evidence (log) and within-kind posterior prediction."""
    log_z = pt.logsumexp(log_joint, axis=1)
    post = pt.exp(log_joint - log_z[:, None])
    return log_z, pred_fn(post)


with pm.Model() as model:
    fam_ll = pm.Data("fam_ll", np.zeros((1, N_FAMILIAR)))  # (sets, rules)
    dsum_ll = pm.Data("dsum_ll", np.zeros((1, 18)))
    range_log_total = pm.Data("range_log_total", np.zeros((1, N_GROUPS)))  # (sets, groups)
    range_in_share = pm.Data("range_in_share", np.zeros((1, N_GROUPS, 100)))
    row_set = pm.Data("row_set", np.zeros(1, dtype="int64"))
    row_target = pm.Data("row_target", np.zeros(1, dtype="int64"))
    row_part = pm.Data("row_part", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    interval_scale = pm.LogNormal("interval_scale", mu=np.log(20.0), sigma=1.0)
    dsum_decay = pm.Normal("dsum_decay", mu=0.0, sigma=0.5)
    round_bonus = pm.Normal("round_bonus", mu=0.0, sigma=1.0)
    # Population mean log-odds of digit-sum rules and of ranges vs familiar
    # rules, and how much participants differ around it (non-centred).
    kind_mu = pm.Normal("kind_mu", mu=0.0, sigma=2.0, shape=2)
    kind_sd = pm.HalfNormal("kind_sd", sigma=1.0)
    kind_z = pm.Normal("kind_z", mu=0.0, sigma=1.0, shape=N_PART_MAX)
    logits = pt.concatenate([pt.zeros((N_PART_MAX, 1)), pt.zeros((N_PART_MAX, 1)) + kind_mu[0], kind_mu[1] + kind_sd * kind_z[:, None]], axis=1)
    log_w = logits - pt.logsumexp(logits, axis=1, keepdims=True)  # (participants, 3)

    # Within-kind evidence and predictions per set (shared by everyone).
    z_fam, p_fam = cond_kind(fam_ll - np.log(N_FAMILIAR), lambda q: q @ pt.as_tensor(FAM_MEM))
    dsum_share = -dsum_decay * DSUM_VALUE
    dsum_share = dsum_share - pt.logsumexp(dsum_share)
    z_dsum, p_dsum = cond_kind(dsum_ll + dsum_share[None, :], lambda q: q @ pt.as_tensor(DSUM_MEM))
    raw = np.log(GROUP_WIDTH) - GROUP_WIDTH / interval_scale + round_bonus * GROUP_EDGES
    log_per_range = raw - pt.logsumexp(raw + np.log(np.maximum(GROUP_COUNT, 1e-300)))
    z_rng, p_rng = cond_kind(log_per_range[None, :] + range_log_total, lambda q: pt.batched_dot(q[:, None, :], range_in_share)[:, 0, :])

    # Per trial: the participant's kind posterior mixes the within-kind predictions.
    log_z = pt.stack([z_fam, z_dsum, z_rng], axis=1)[row_set] + log_w[row_part]  # (rows, 3)
    kind_post = pt.exp(log_z - pt.logsumexp(log_z, axis=1, keepdims=True))
    preds = pt.stack(
        [p_fam[row_set, row_target], p_dsum[row_set, row_target], p_rng[row_set, row_target]], axis=1
    )
    p_yes = pm.Deterministic("p_yes", pt.clip((kind_post * preds).sum(axis=1), 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = np.array([int(r["target"]) for r in rows])
    trial = np.array([int(float(r.get("trial_index", 0) or 0)) for r in rows])
    part = np.concatenate([[0], np.cumsum(np.diff(trial) < 0)]) if len(rows) else np.zeros(0)
    part = np.minimum(part, N_PART_MAX - 1).astype("int64")
    unique = list(dict.fromkeys(sets))
    lookup = {s: i for i, s in enumerate(unique)}
    index = np.array([lookup[s] for s in sets])
    total = np.zeros((len(unique), N_GROUPS))
    inside = np.zeros((len(unique), N_GROUPS, 100))
    for i, s in enumerate(unique):
        ll = np.zeros(len(LO))
        for x in s:
            ll += np.where((LO <= x) & (x <= HI), LOG_IN[WIDTH - 1], LOG_OUT)
        top = ll.max()
        e = np.exp(ll - top)
        tot = np.bincount(GROUP, weights=e, minlength=N_GROUPS)
        total[i] = top + np.log(np.maximum(tot, 1e-300))
        for t in range(1, 101):
            hit = (LO <= t) & (t <= HI)
            inside[i, :, t - 1] = np.bincount(GROUP[hit], weights=e[hit], minlength=N_GROUPS) / np.maximum(tot, 1e-300)
    return {
        "fam_ll": np.stack([FAM_LL[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "dsum_ll": np.stack([DSUM_LL[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "range_log_total": total,
        "range_in_share": inside,
        "row_set": index.astype("int64"),
        "row_target": (targets - 1).astype("int64"),
        "row_part": part,
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
