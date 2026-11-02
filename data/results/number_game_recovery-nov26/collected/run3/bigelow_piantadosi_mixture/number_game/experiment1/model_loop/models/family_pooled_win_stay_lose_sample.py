"""Refinement of win_stay_lose_sample_concepts: people learn the concept one
example at a time, holding a single working guess rather than the full
posterior. After the first example they pick a rule or magnitude interval from
the Bayesian posterior (the 45 Bigelow-Piantadosi rules and every interval of
1-100, size principle, 10% of examples drawn uniformly); as each later example
arrives they keep the current guess with probability `stay` if it still
contains that example, and otherwise re-pick from the posterior given all
examples so far. "Yes" is the probability that the final guess contains the
target, so the order of the examples matters.

The one change (grafted from family_pooled_rule_prior) is the prior over rules
from which guesses are picked: instead of an independent weight for every
rule, each rule's log prior weight is drawn around its family's level (parity,
primes, squares and cubes, powers, multiples, final digit, contained digit)
with an inferred spread tau (partial pooling, non-centred), so rules of the
same kind are similarly likely to be picked as the working guess.

The marginal distribution of the guess is tracked exactly. Write P_t for the
posterior after the first t examples, R_t for the probability of re-sampling at
step t and c_t(h) for "h contains example t". Then
  D_t(h) = stay * c_t(h) * D_{t-1}(h) + R_t * P_t(h),
  R_t = 1 - stay * sum_h c_t(h) D_{t-1}(h).
With the example-noise rate fixed, interval sums are precomputed in
prepare_observed. Sets shorter than four examples are padded with dummy steps
that every concept contains and at which nobody re-samples.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_N = 4
ALPHA = 0.9  # probability an example is drawn from the concept


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


_RULES = rule_concepts()
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
INT_LO = np.array([m for m in NUMBERS for n in range(m, 101)])
INT_HI = np.array([n for m in NUMBERS for n in range(m, 101)])
INT_SIZE = (INT_HI - INT_LO + 1).astype("float64")
LOG_NOISE = np.log(1 - ALPHA) - np.log(100)
LOG_IN_RULE = np.log(ALPHA / RULE_SIZE + (1 - ALPHA) / 100)
LOG_IN_INT = np.log(ALPHA / INT_SIZE + (1 - ALPHA) / 100)

FAMILIES = ["parity", "primes", "squares and cubes", "powers", "multiples", "ends in", "contains digit"]


def family_of(name):
    if name in ("odd", "even"):
        return "parity"
    if name in ("squares", "cubes"):
        return "squares and cubes"
    for f in ("primes", "powers", "multiples", "ends in", "contains digit"):
        if name.startswith(f):
            return f
    raise ValueError(name)


RULE_FAMILY = np.array([FAMILIES.index(family_of(r)) for r in RULE_NAMES])

with pm.Model() as model:
    rule_k = pm.Data("rule_k", np.zeros((1, MAX_N, N_RULES)))  # examples in rule among prefix t
    pref_n = pm.Data("pref_n", np.ones((1, MAX_N)))  # prefix length (padded steps repeat n)
    rule_c = pm.Data("rule_c", np.ones((1, MAX_N, N_RULES)))  # rule contains example t (1 if dummy)
    active = pm.Data("active", np.ones((1, MAX_N)))  # step t is a real example
    int_ev = pm.Data("int_ev", np.zeros((1, MAX_N)))  # log sum of interval likelihoods, prefix t
    frac_cons = pm.Data("frac_cons", np.ones((1, MAX_N, MAX_N)))  # [t', t]: share containing ex t'+1..t
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    cell_frac = pm.Data("cell_frac", np.zeros((1, MAX_N)))  # [t']: share containing later ex and target
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    # Each rule's prior log-weight is drawn around its family's level, with an
    # inferred spread (non-centred): tau = 0 gives equal weights within a family.
    family_level = pm.Normal("family_level", mu=0.0, sigma=1.5, shape=len(FAMILIES))
    tau = pm.HalfNormal("tau", sigma=1.0)
    rule_dev = pm.Normal("rule_dev", mu=0.0, sigma=1.0, shape=N_RULES)
    rule_weights = pm.Deterministic(
        "rule_weights", pt.special.softmax(family_level[RULE_FAMILY] + tau * rule_dev)
    )
    # Probability of keeping a guess that is still consistent with a new example.
    stay_logit = pm.Normal("stay_logit", mu=0.0, sigma=1.5)
    stay = pm.Deterministic("stay", pm.math.sigmoid(stay_logit))

    # Posterior after each prefix: rules (sets, t, rules), intervals as a group (sets, t).
    log_rules = (pt.log(lambda_0) + pt.log(rule_weights)[None, None, :]
                 + rule_k * LOG_IN_RULE[None, None, :] + (pref_n[:, :, None] - rule_k) * LOG_NOISE)
    log_int = pt.log1p(-lambda_0) - 2 * np.log(100) + int_ev
    log_z = pt.logaddexp(pt.logsumexp(log_rules, axis=2), log_int)
    p_rule = pt.exp(log_rules - log_z[:, :, None])
    q_int = pt.exp(log_int - log_z)

    s_step = active * stay + (1 - active)  # padded steps: everyone stays
    d_rule = p_rule[:, 0, :]
    gains = [pt.ones_like(q_int[:, 0])]  # gains[t'] = R_t' * prod_{u in t'+1..t} s_u
    for t in range(1, MAX_N):
        consistent = (d_rule * rule_c[:, t, :]).sum(axis=1)
        for tp in range(t):
            consistent = consistent + gains[tp] * q_int[:, tp] * frac_cons[:, tp, t]
        r_t = active[:, t] * pt.clip(1 - s_step[:, t] * consistent, 0.0, 1.0)
        d_rule = s_step[:, t, None] * rule_c[:, t, :] * d_rule + r_t[:, None] * p_rule[:, t, :]
        gains = [g * s_step[:, t] for g in gains] + [r_t]

    gain_mat = pt.stack(gains, axis=1)  # (sets, t')
    in_concept = ((d_rule[cell_set] * cell_rule_member).sum(axis=1)
                  + (gain_mat[cell_set] * q_int[cell_set] * cell_frac).sum(axis=1))
    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept, 1e-6, 1 - 1e-6)[row_cell])
    pm.Bernoulli("response", p=p_yes, observed=rating)


def _logsumexp(x):
    m = x.max()
    return m + np.log(np.exp(x - m).sum())


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    rule_k = np.zeros((S, MAX_N, N_RULES))
    pref_n = np.zeros((S, MAX_N))
    rule_c = np.ones((S, MAX_N, N_RULES))
    active = np.zeros((S, MAX_N))
    int_ev = np.zeros((S, MAX_N))
    frac_cons = np.ones((S, MAX_N, MAX_N))
    int_ll = {}  # (set, t') -> per-interval log likelihood
    later_in = {}  # (set, t') -> intervals containing every real example after t'
    for i, s in enumerate(unique):
        ex = np.array(s)
        n = len(ex)
        inside = (ex[None, :] >= INT_LO[:, None]) & (ex[None, :] <= INT_HI[:, None])  # (intervals, n)
        for t in range(MAX_N):
            m = min(t + 1, n)
            active[i, t] = 1.0 if t < n else 0.0
            pref_n[i, t] = m
            rule_k[i, t] = RULE_MEMBERSHIP[:, ex[:m] - 1].sum(axis=1)
            if t < n:
                rule_c[i, t] = RULE_MEMBERSHIP[:, ex[t] - 1]
            k = inside[:, :m].sum(axis=1)
            ll = k * LOG_IN_INT + (m - k) * LOG_NOISE
            int_ev[i, t] = _logsumexp(ll)
            w = np.exp(ll - int_ev[i, t])
            int_ll[i, t] = w
            for t2 in range(t + 1, MAX_N):
                cont = inside[:, t + 1:min(t2 + 1, n)].all(axis=1)
                frac_cons[i, t, t2] = w[cont].sum()
            later_in[i, t] = inside[:, t + 1:n].all(axis=1)
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: j for j, c in enumerate(cells)}
    cell_set = np.array([i for i, _ in cells], dtype="int64")
    cell_rule_member = np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells])
    cell_frac = np.zeros((len(cells), MAX_N))
    for j, (i, tg) in enumerate(cells):
        has_t = (INT_LO <= tg) & (tg <= INT_HI)
        for t in range(MAX_N):
            cell_frac[j, t] = int_ll[i, t][has_t & later_in[i, t]].sum()
    return {
        "rule_k": rule_k, "pref_n": pref_n, "rule_c": rule_c, "active": active,
        "int_ev": int_ev, "frac_cons": frac_cons, "cell_set": cell_set,
        "cell_rule_member": cell_rule_member, "cell_frac": cell_frac,
        "row_cell": np.array([cell_index[c] for c in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
