"""Volatile-concept changepoint updating in which ranges, not just rules, are
perceived as inflated in size — by their own amount, replacing range tempering.

Refinement of volatile_changepoint_inflated_size: people read the examples one
at a time, updating a Bayesian belief over 42 weighted rules (rule-size prior)
and all intervals of 1..100 while allowing, with an inferred switch
probability, that the program re-draws its concept between outputs (10% noisy
examples, first-example evidence multiplier, decision gain and shifted
criterion), and in the size principle they perceive concepts as larger than
they are. The one change is how the size principle is weakened for magnitude
ranges: instead of the range evidence being discounted by its own tempering
weight (a power on the likelihood) while ranges share the rules' inflation, a
range's perceived width is its true width plus its own inferred number of extra
members, and one evidence weight serves rules and ranges alike. So each
example's in-concept likelihood is 1 / (size + rule_inflation) for rules and
1 / (width + range_inflation) for ranges: the evidential penalty for wide
ranges saturates — narrow ranges around the examples lose their edge over
moderately wider ones, while very wide ranges are still penalised.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

EPSILON = 0.1
NUMBERS = np.arange(1, 101)
MAX_SET = 4


def concept_sets():
    """42 rules, then 5,050 intervals."""
    rules = [
        [2, 4, 8, 16, 32, 64], [3, 9, 27, 81], [4, 16, 64], [5, 25], [6, 36],
        [7, 49], [8, 64], [9, 81], [10, 100],
    ]
    rules += [[i for i in NUMBERS if i % n == 0] for n in range(3, 13)]
    rules += [[i for i in NUMBERS if i % 10 == d] for d in range(1, 10)]
    rules += [[i for i in NUMBERS if str(d) in str(i)] for d in range(1, 10)]
    rules.append([1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97])
    rules.append([i for i in NUMBERS if i % 2 == 0])
    rules.append([i for i in NUMBERS if i % 2 == 1])
    rules.append([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])
    rules.append([1, 8, 27, 64])
    intervals = [list(range(m, n + 1)) for m in range(1, 101) for n in range(m, 101)]
    return rules, intervals


RULES, INTERVALS = concept_sets()
N_RULES = len(RULES)
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]
INTERVAL_WIDTH = INTERVAL_MEMBERSHIP.sum(axis=1).astype("int64")
LOG_N_INTERVALS = np.log(len(INTERVALS))
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
RULE_LOG_SIZE = np.log(RULE_SIZE)
RULE_LOG_SIZE_C = RULE_LOG_SIZE - RULE_LOG_SIZE.mean()
LOG_NOISE = np.log(EPSILON / 100)

# Sets are processed in buckets by their number of examples (1..4), so each
# bucket runs exactly as many sequential updates as it has examples. Within a
# set, intervals of the same width containing the same examples share a prior
# and a whole sequential likelihood trajectory, so prepare_observed collapses
# them into groups; the collapsing is exact. A constant log-likelihood shared by
# every concept at a step (the noise term) cancels in normalisation and is
# dropped.

with pm.Model() as model:
    buckets = []
    for n in range(1, MAX_SET + 1):
        buckets.append(
            dict(
                n=n,
                rule_in=pm.Data(f"b{n}_rule_in", np.zeros((1, n, N_RULES))),  # (sets, step, rules)
                group_in=pm.Data(f"b{n}_group_in", np.zeros((1, n, 2))),  # (sets, step, groups)
                width=pm.Data(f"b{n}_width", np.ones((1, 2))),  # (sets, groups)
                log_count=pm.Data(f"b{n}_log_count", np.zeros((1, 2))),  # (sets, groups)
                share=pm.Data(f"b{n}_share", np.zeros((1, 2))),  # (cells, groups)
                cell_set=pm.Data(f"b{n}_cell_set", np.zeros(1, dtype="int64")),
                cell_flat=pm.Data(f"b{n}_cell_flat", np.zeros(1, dtype="int64")),  # set * 100 + target - 1
            )
        )
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # trial -> cell (buckets concatenated)
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 4, 4)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    evidence_weight = pm.LogNormal("evidence_weight", mu=0.0, sigma=0.5)
    log_gain = pm.Normal("log_gain", mu=0.0, sigma=0.5)
    bias = pm.Normal("bias", mu=0.0, sigma=1.0)
    anchor_weight = pm.LogNormal("anchor_weight", mu=0.0, sigma=0.75)
    size_exponent = pm.Normal("size_exponent", mu=0.0, sigma=0.5)
    switch_prob = pm.Beta("switch_prob", 1.5, 6.0)
    # Perceived size = true size + inflation; the refinement gives ranges their
    # own inflation in place of their own evidence tempering.
    rule_inflation = pm.HalfNormal("rule_inflation", sigma=10.0)
    range_inflation = pm.HalfNormal("range_inflation", sigma=10.0)

    def member_gain(size, inflation):
        """Evidence of a consistent example relative to a noise-explained one."""
        return evidence_weight * (pt.log((1 - EPSILON) / (size + inflation) + EPSILON / 100) - LOG_NOISE)

    rule_gain = member_gain(RULE_SIZE, rule_inflation)  # (rules,)
    log_rule_prior = pt.log(rule_weights) + size_exponent * RULE_LOG_SIZE_C
    log_rule_prior = pt.log(p_mix) + log_rule_prior - pt.logsumexp(log_rule_prior)
    log_stay = pt.log1p(-switch_prob)
    log_switch = pt.log(switch_prob)
    rule_matrix = pt.as_tensor(RULE_MEMBERSHIP)

    cell_probs = []
    for b in buckets:
        n_sets = b["log_count"].shape[0]
        log_prior = pt.concatenate(
            [
                pt.broadcast_to(log_rule_prior[None, :], (n_sets, N_RULES)),
                pt.log1p(-p_mix) - LOG_N_INTERVALS + b["log_count"],
            ],
            axis=1,
        )  # (sets, concepts)
        group_gain = member_gain(b["width"], range_inflation)  # (sets, groups)
        step_ev = pt.concatenate(
            [b["rule_in"] * rule_gain[None, None, :], b["group_in"] * group_gain[:, None, :]], axis=2
        )  # (sets, step, concepts)
        log_post = log_prior + anchor_weight * step_ev[:, 0, :]
        log_post = log_post - pt.logsumexp(log_post, axis=1, keepdims=True)
        for k in range(1, b["n"]):
            relaxed = pt.logaddexp(log_stay + log_post, log_switch + log_prior)
            log_post = relaxed + step_ev[:, k, :]
            log_post = log_post - pt.logsumexp(log_post, axis=1, keepdims=True)
        posterior = pt.exp(log_post)
        rule_part = (posterior[:, :N_RULES] @ rule_matrix).reshape((-1,))[b["cell_flat"]]
        interval_part = (posterior[:, N_RULES:][b["cell_set"]] * b["share"]).sum(axis=1)
        cell_probs.append(rule_part + interval_part)

    concept_p = pt.clip(pt.concatenate(cell_probs), 1e-6, 1 - 1e-6)
    belief_logit = pt.log(concept_p) - pt.log1p(-concept_p)
    cell_p = pt.clip(pm.math.sigmoid(pt.exp(log_gain) * belief_logit + bias), 1e-6, 1 - 1e-6)
    p_yes = pm.Deterministic("p_yes", cell_p[row_cell])
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = np.array([int(r["target"]) - 1 for r in rows], dtype="int64")
    out = {}
    cell_offset = 0
    row_cell = np.zeros(len(rows), dtype="int64")
    for n in range(1, MAX_SET + 1):
        unique = list(dict.fromkeys(s for s in sets if len(s) == n))
        index = {s: i for i, s in enumerate(unique)}
        per_set = []
        for s in unique:
            ex = np.array(s) - 1
            inside = INTERVAL_MEMBERSHIP[:, ex].astype("int64")  # (intervals, n)
            pattern = (inside * (2 ** np.arange(n)))[:, :].sum(axis=1)
            group = (INTERVAL_WIDTH - 1) * (2 ** MAX_SET) + pattern
            per_set.append((ex, np.unique(group, return_inverse=True)))
        n_sets = max(len(unique), 1)
        n_groups = max([len(u) for _, (u, _) in per_set] + [2])
        log_count = np.full((n_sets, n_groups), -700.0)
        group_in = np.zeros((n_sets, n, n_groups))
        width = np.ones((n_sets, n_groups))
        share = np.zeros((n_sets, n_groups, 100))
        rule_in = np.zeros((n_sets, n, N_RULES))
        for i, (ex, (u, inv)) in enumerate(per_set):
            inv = inv.reshape(-1)
            counts = np.bincount(inv, minlength=len(u))
            log_count[i, : len(u)] = np.log(counts)
            first = np.zeros(len(u), dtype="int64")
            first[inv[::-1]] = np.arange(len(inv))[::-1]  # a representative interval per group
            width[i, : len(u)] = INTERVAL_WIDTH[first]
            for k, e in enumerate(ex):
                group_in[i, k, : len(u)] = INTERVAL_MEMBERSHIP[first, e]
                rule_in[i, k] = RULE_MEMBERSHIP[:, e]
            np.add.at(share[i], inv, INTERVAL_MEMBERSHIP)
            share[i, : len(u)] /= counts[:, None]
        in_bucket = np.array([len(s) == n for s in sets])
        set_idx = np.array([index[s] for s, m in zip(sets, in_bucket) if m], dtype="int64")
        flat = set_idx * 100 + targets[in_bucket]
        cells, local_cell = np.unique(flat, return_inverse=True)
        row_cell[in_bucket] = cell_offset + local_cell.reshape(-1)
        cell_offset += len(cells)
        c_set, c_tgt = cells // 100, cells % 100
        out.update(
            {
                f"b{n}_rule_in": rule_in,
                f"b{n}_group_in": group_in,
                f"b{n}_width": width,
                f"b{n}_log_count": log_count,
                f"b{n}_share": share[c_set, :, c_tgt],
                f"b{n}_cell_set": c_set.astype("int64"),
                f"b{n}_cell_flat": cells.astype("int64"),
            }
        )
    out["row_cell"] = row_cell
    out["rating"] = np.array([int(r["rating"]) for r in rows], dtype="int64")
    return out
