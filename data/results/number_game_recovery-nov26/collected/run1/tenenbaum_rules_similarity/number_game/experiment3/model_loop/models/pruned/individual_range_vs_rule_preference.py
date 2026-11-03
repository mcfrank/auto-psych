"""Individual kind preferences in Bayesian concept learning.

Refinement of `equal_prior_rules_round_edge_ranges`: every participant does the
same Bayesian concept learning over equally plausible familiar number rules,
digit-sum rules (prior share changing smoothly with the sum) and numeric ranges
(Erlang width prior, round-number range edges, the size principle, 10% of
examples uniform noise). The one change is that people are not all alike in how
readily they think of the program as a magnitude range rather than a number
rule: each participant has their own prior log-odds of ranges versus rules
(the split between familiar rules and digit sums is shared). So some people are
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


with pm.Model() as model:
    # Per distinct (set, target) pair, all precomputed from the stimulus:
    # familiar rules (no free parameters): log evidence and P(target | rules).
    pair_fam_logz = pm.Data("pair_fam_logz", np.zeros(1))
    pair_fam_pred = pm.Data("pair_fam_pred", np.zeros(1))
    # Digit sums: per sum, exp(log-likelihood of the set - its maximum), the
    # maximum, and the same times whether the target has that sum.
    pair_dsum_lik = pm.Data("pair_dsum_lik", np.zeros((1, 18)))
    pair_dsum_top = pm.Data("pair_dsum_top", np.zeros(1))
    pair_dsum_hit = pm.Data("pair_dsum_hit", np.zeros((1, 18)))
    # Ranges: per width/round-edge group, the summed scaled likelihood of the
    # set over the group's ranges, its scale, and the part from ranges that
    # contain the target.
    pair_range_lik = pm.Data("pair_range_lik", np.zeros((1, N_GROUPS)))
    pair_range_top = pm.Data("pair_range_top", np.zeros(1))
    pair_range_hit = pm.Data("pair_range_hit", np.zeros((1, N_GROUPS)))
    row_pair = pm.Data("row_pair", np.zeros(1, dtype="int64"))
    row_part_onehot = pm.Data("row_part_onehot", np.zeros((1, N_PART_MAX)))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    interval_scale = pm.LogNormal("interval_scale", mu=np.log(20.0), sigma=1.0)
    dsum_decay = pm.Normal("dsum_decay", mu=0.0, sigma=0.5)
    round_bonus = pm.Normal("round_bonus", mu=0.0, sigma=1.0)
    # Shared prior log-odds of digit-sum rules vs familiar rules, and each
    # participant's own prior log-odds of ranges vs familiar rules.
    dsum_logit = pm.Normal("dsum_logit", mu=0.0, sigma=2.0)
    range_logit = pm.Normal("range_logit", mu=0.0, sigma=2.0, shape=N_PART_MAX)

    # Digit sums: prior share changing smoothly with the sum.
    dsum_share = -dsum_decay * DSUM_VALUE
    dsum_share = dsum_share - pt.logsumexp(dsum_share)
    dsum_top = pt.max(dsum_share)
    dsum_w = pt.exp(dsum_share - dsum_top)
    dsum_den = pt.dot(pair_dsum_lik, dsum_w)
    z_dsum = pair_dsum_top + dsum_top + pt.log(dsum_den)
    pred_dsum = pt.dot(pair_dsum_hit, dsum_w) / dsum_den
    # Ranges: Erlang width prior with round-number edges, per range.
    raw = np.log(GROUP_WIDTH) - GROUP_WIDTH / interval_scale + round_bonus * GROUP_EDGES
    log_per_range = raw - pt.logsumexp(raw + np.log(np.maximum(GROUP_COUNT, 1e-300)))
    rng_top = pt.max(log_per_range)
    rng_w = pt.exp(log_per_range - rng_top)
    rng_den = pt.dot(pair_range_lik, rng_w)
    z_rng = pair_range_top + rng_top + pt.log(rng_den)
    pred_rng = pt.dot(pair_range_hit, rng_w) / rng_den

    pair_stats = pt.stack(
        [z_rng - pair_fam_logz, z_dsum - pair_fam_logz, pair_fam_pred, pred_dsum, pred_rng], axis=1
    )[row_pair]  # (rows, 5)

    # Per trial: the participant's prior odds of ranges set the kind posterior,
    # which mixes the within-kind predictions.
    row_range_logit = pt.dot(row_part_onehot, range_logit)
    log_k = pt.stack(
        [pt.zeros_like(row_range_logit), pair_stats[:, 1] + dsum_logit, pair_stats[:, 0] + row_range_logit], axis=1
    )
    kind_post = pt.exp(log_k - pt.logsumexp(log_k, axis=1, keepdims=True))
    p_yes = pm.Deterministic("p_yes", pt.clip((kind_post * pair_stats[:, 2:]).sum(axis=1), 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def logsumexp_rows(x):
    top = x.max(axis=1, keepdims=True)
    return (top + np.log(np.exp(x - top).sum(axis=1, keepdims=True)))[:, 0]


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    trial = np.array([int(float(r.get("trial_index", 0) or 0)) for r in rows])
    part = np.concatenate([[0], np.cumsum(np.diff(trial) < 0)]) if len(rows) else np.zeros(0)
    part = np.minimum(part, N_PART_MAX - 1).astype("int64")
    unique = list(dict.fromkeys(sets))
    set_id = {s: i for i, s in enumerate(unique)}
    pairs = list(dict.fromkeys(zip(sets, targets)))
    pair_id = {p: i for i, p in enumerate(pairs)}
    range_sets = []
    for s in unique:
        ll = np.zeros(len(LO))
        for x in s:
            ll += np.where((LO <= x) & (x <= HI), LOG_IN[WIDTH - 1], LOG_OUT)
        top = ll.max()
        e = np.exp(ll - top)
        range_sets.append((top, e, np.bincount(GROUP, weights=e, minlength=N_GROUPS)))
    n = len(pairs)
    out = {k: np.zeros(n) for k in ["pair_fam_logz", "pair_fam_pred", "pair_dsum_top", "pair_range_top"]}
    out["pair_dsum_lik"], out["pair_dsum_hit"] = np.zeros((n, 18)), np.zeros((n, 18))
    out["pair_range_lik"], out["pair_range_hit"] = np.zeros((n, N_GROUPS)), np.zeros((n, N_GROUPS))
    for j, (s, t) in enumerate(pairs):
        idx = np.array(s) - 1
        fam = FAM_LL[:, idx].sum(axis=1) - np.log(N_FAMILIAR)
        out["pair_fam_logz"][j] = logsumexp_rows(fam[None, :])[0]
        q = np.exp(fam - out["pair_fam_logz"][j])
        out["pair_fam_pred"][j] = float(q @ FAM_MEM[:, t - 1])
        dll = DSUM_LL[:, idx].sum(axis=1)
        out["pair_dsum_top"][j] = dll.max()
        out["pair_dsum_lik"][j] = np.exp(dll - dll.max())
        out["pair_dsum_hit"][j] = out["pair_dsum_lik"][j] * DSUM_MEM[:, t - 1]
        top, e, tot = range_sets[set_id[s]]
        hit = (LO <= t) & (t <= HI)
        out["pair_range_top"][j] = top
        out["pair_range_lik"][j] = tot
        out["pair_range_hit"][j] = np.bincount(GROUP[hit], weights=e[hit], minlength=N_GROUPS)
    out["row_pair"] = np.array([pair_id[p] for p in zip(sets, targets)], dtype="int64")
    out["row_part_onehot"] = np.eye(N_PART_MAX)[part]
    out["rating"] = np.array([int(r["rating"]) for r in rows], dtype="int64")
    return out
