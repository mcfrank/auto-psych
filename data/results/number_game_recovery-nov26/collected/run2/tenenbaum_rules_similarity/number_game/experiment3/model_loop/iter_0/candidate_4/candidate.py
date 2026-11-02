"""Refinement of no_digit_containment_span_concepts: the same Bayesian concept
learning (arithmetic and final-digit rules plus multiples of 13-50,
span-restricted runs of the base rules, ranges with an Erlang width prior, an
inferred share of random examples, examples drawn without replacement, no
"contains digit d" concepts), except that the prior over ranges also depends on
where a range sits on the number line: its log prior shifts linearly with the
range's midpoint, with strength and sign inferred (zero recovers the incumbent).
This predicts that "yes" extends further below examples that fit no rule than
above them (or vice versa), the up/down extrapolation asymmetry.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_EX = 4


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
    return rules


_RULES = rule_concepts()
N_BASE = len(_RULES)
BASE_MEMBERS = [sorted(c) for c in _RULES.values()]
EXT_M = list(range(13, 51))
N_EXT = len(EXT_M)
_RULES.update({f"multiples of {m}": list(range(m, 101, m)) for m in EXT_M})
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)

# Span-restricted rules: runs of 2..size-1 consecutive members of a base rule,
# grouped by (run length s, examples inside k), pooled over rules.
MAX_RUN = max(len(m) for m in BASE_MEMBERS) - 1
SG_SIZE = np.repeat(np.arange(2, MAX_RUN + 1), MAX_EX + 1).astype("float64")
SG_NIN = np.tile(np.arange(MAX_EX + 1), MAX_RUN - 1).astype("float64")
N_SG = len(SG_SIZE)
N_RUNS_TOTAL = float(sum(len(m) - s + 1 for m in BASE_MEMBERS for s in range(2, len(m))))

# Every interval [lo, hi] of 1..100, listed once; likelihoods are looked up
# per set from a (size, examples inside) table.
SIZES = np.arange(1, 101)
GROUP_SIZE = np.repeat(SIZES, MAX_EX + 1).astype("float64")
GROUP_NIN = np.tile(np.arange(MAX_EX + 1), 100).astype("float64")
N_GROUPS = len(GROUP_SIZE)
IV_LO, IV_HI = np.array([(lo, hi) for lo in range(1, 101) for hi in range(lo, 101)]).T
N_IV = len(IV_LO)
IV_SIZE = (IV_HI - IV_LO + 1).astype("float64")
# Midpoint of the range, centred and scaled to [-1, 1].
IV_MID = ((IV_LO + IV_HI) / 2.0 - 50.5) / 49.5
IV_CONTAINS = ((IV_LO[:, None] <= NUMBERS[None, :]) & (IV_HI[:, None] >= NUMBERS[None, :])).astype("float64")

RULE_CONCENTRATION = np.full(N_BASE, 2.0)


def _wor_loglik(size, n_in, n_examples, noise):
    """Without-replacement log-likelihood for concepts of given size with n_in examples inside."""
    log_out = pt.log(noise / 100.0)
    ll = (n_examples[:, None] - n_in[None, :]) * log_out
    for j in range(MAX_EX):
        term = pt.log((1 - noise) / np.maximum(size - j, 1.0) + noise / 100.0)
        ll = ll + ((n_in > j) * term)[None, :]
    return ll


with pm.Model() as model:
    rule_n_in = pm.Data("rule_n_in", np.zeros((1, N_RULES)))
    n_examples = pm.Data("n_examples", np.ones(1))
    iv_group = pm.Data("iv_group", np.zeros((1, N_IV), dtype="int64"))
    group_present = pm.Data("group_present", np.zeros((1, N_GROUPS)))
    span_log_count = pm.Data("span_log_count", np.zeros((1, N_SG)))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_target = pm.Data("cell_target", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    cell_span_frac = pm.Data("cell_span_frac", np.zeros((1, N_SG)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    noise = pm.Beta("noise", 1.5, 6.0)
    rule_weights = pm.Dirichlet("rule_weights", a=RULE_CONCENTRATION)
    ext_share = pm.Beta("ext_share", 1.0, 4.0)
    span_share = pm.Beta("span_share", 1.0, 4.0)
    all_weights = pt.concatenate([
        (1 - ext_share) * rule_weights, pt.ones(N_EXT) * ext_share / N_EXT,
    ])
    log_span_prior = pt.log(lambda_0) + pt.log(span_share) - np.log(N_RUNS_TOTAL)

    sigma = pm.LogNormal("sigma", mu=np.log(20.0), sigma=1.0)
    # The one change: a range's log prior shifts linearly with its midpoint
    # (positive = ranges among small numbers favoured; zero = incumbent).
    location_bias = pm.Normal("location_bias", mu=0.0, sigma=2.0)
    log_iv_prior = np.log(IV_SIZE) - IV_SIZE / sigma - location_bias * IV_MID
    log_iv_prior = pt.log1p(-lambda_0) + log_iv_prior - pt.logsumexp(log_iv_prior)

    log_out = pt.log(noise / 100.0)
    rule_log_lik = (n_examples[:, None] - rule_n_in) * log_out
    for j in range(MAX_EX):
        rule_term = pt.log((1 - noise) / np.maximum(RULE_SIZE - j, 1.0) + noise / 100.0)
        rule_log_lik = rule_log_lik + pt.gt(rule_n_in, j) * rule_term[None, :]
    group_log_lik = _wor_loglik(GROUP_SIZE, GROUP_NIN, n_examples, noise)
    span_log_lik = _wor_loglik(SG_SIZE, SG_NIN, n_examples, noise)

    log_rules = pt.log(lambda_0) + pt.log1p(-span_share) + pt.log(all_weights)[None, :] + rule_log_lik
    log_spans = log_span_prior + span_log_count + span_log_lik

    # Interval part in probability space, shifted by a per-set bound for stability.
    present_lik = group_log_lik + (group_present - 1.0) * 1e4
    shift = pt.max(present_lik, axis=1, keepdims=True) + pt.max(log_iv_prior)
    rel_lik = pt.exp(group_log_lik - shift)
    iv_weight = rel_lik.ravel()[iv_group] * pt.exp(log_iv_prior)[None, :]
    iv_mass = iv_weight.sum(axis=1, keepdims=True)
    in_range_mass = pt.dot(iv_weight, IV_CONTAINS)

    log_z = pt.logsumexp(pt.stack([
        pt.logsumexp(log_rules, axis=1), pt.log(iv_mass[:, 0]) + shift[:, 0],
        pt.logsumexp(log_spans, axis=1),
    ], axis=1), axis=1, keepdims=True)
    post_rules = pt.exp(log_rules - log_z)
    post_spans = pt.exp(log_spans - log_z)
    in_range = in_range_mass * pt.exp(shift - log_z)
    in_concept = (
        (post_rules[cell_set] * cell_rule_member).sum(axis=1)
        + in_range[cell_set, cell_target]
        + (post_spans[cell_set] * cell_span_frac).sum(axis=1)
    )

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-9, 1 - 1e-9))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set`, `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    rule_n_in = np.zeros((S, N_RULES))
    n_ex = np.zeros(S)
    iv_group = np.zeros((S, N_IV), dtype="int64")
    group_present = np.zeros((S, N_GROUPS))
    scounts = np.zeros((S, N_SG))
    scontain = np.zeros((S, N_SG, 100))
    for i, s in enumerate(unique):
        ex = np.array(s) - 1
        n_ex[i] = len(s)
        rule_n_in[i] = RULE_MEMBERSHIP[:, ex].sum(axis=1)
        k = np.minimum(((IV_LO[:, None] <= ex[None, :] + 1) & (IV_HI[:, None] >= ex[None, :] + 1)).sum(axis=1), MAX_EX)
        g = (IV_SIZE.astype("int64") - 1) * (MAX_EX + 1) + k
        iv_group[i] = i * N_GROUPS + g
        group_present[i, np.unique(g)] = 1.0
        for mem in BASE_MEMBERS:
            L = len(mem)
            is_ex = np.array([sum(1 for x in s if x == v) for v in mem])
            cum = np.concatenate([[0], np.cumsum(is_ex)])
            for size in range(2, L):
                base = (size - 2) * (MAX_EX + 1)
                for a in range(0, L - size + 1):
                    kk = int(min(cum[a + size] - cum[a], MAX_EX))
                    g = base + kk
                    scounts[i, g] += 1
                    scontain[i, g, np.array(mem[a:a + size]) - 1] += 1
    sfrac = np.where(scounts[:, :, None] > 0, scontain / np.maximum(scounts, 1)[:, :, None], 0.0)
    slog_count = np.where(scounts > 0, np.log(np.maximum(scounts, 1)), -1e3)
    keys = [(index[s], int(r["target"]) - 1) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {kk: i for i, kk in enumerate(cells)}
    cs = np.array([kk[0] for kk in cells], dtype="int64")
    ct = np.array([kk[1] for kk in cells], dtype="int64")
    return {
        "rule_n_in": rule_n_in,
        "n_examples": n_ex,
        "iv_group": iv_group,
        "group_present": group_present,
        "span_log_count": slog_count,
        "cell_set": cs,
        "cell_target": ct,
        "cell_rule_member": RULE_MEMBERSHIP[:, ct].T.copy(),
        "cell_span_frac": sfrac[cs, :, ct],
        "row_cell": np.array([cell_index[kk] for kk in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
