"""Rule-within-range conjunctive concept learning with a tempered size
principle (refinement of rule_within_range_conjunction).

People infer a conjunctive concept: a mathematical rule (or "any number")
restricted to a range of magnitudes, inferring rule and range together by
Bayesian size-principle reasoning, and answer "yes" with the posterior
probability the concept contains the target. The one change from the
incumbent: people weigh the evidence of the examples with an inferred strength
beta, raising the likelihood of the examples under each hypothesis to the
power beta (beta = 1 is the incumbent). With beta < 1 people are conservative:
the size principle pulls them less sharply toward the tightest rule and range.

Hypotheses are rule r intersected with interval [m, n]; rule prior weights w_r
(Dirichlet); the range is all of 1..100 with probability q, otherwise uniform
over intervals. Examples are drawn from the concept with probability 0.9, else
uniformly from 1..100. Per-set tables are precomputed on a grid of beta values
and linearly interpolated in beta.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

ALPHA = 0.9
NUMBERS = np.arange(1, 101)


def rule_concepts():
    rules = {
        "any number": list(NUMBERS),
        "odd": [n for n in NUMBERS if n % 2 == 1],
        "even": list(range(2, 101, 2)),
        "primes": [n for n in NUMBERS if n > 1 and all(n % a for a in range(2, int(n**0.5) + 1))],
        "squares": [n * n for n in range(1, 11)],
        "cubes": [n**3 for n in range(1, 5)],
        "powers of 2, not 32": [2, 4, 8, 16, 64],
        "powers of 2, or 37": [2, 4, 8, 16, 32, 37, 64],
    }
    rules.update({f"multiples of {m}": list(range(m, 101, m)) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]})
    rules.update({f"powers of {m}": [m**k for k in range(1, 8) if m**k <= 100] for m in range(2, 11)})
    rules.update({f"ends in {d}": [n for n in NUMBERS if n % 10 == d] for d in range(10)})
    # "contains digit 0" equals "ends in 0" on 1..100, so it is omitted.
    rules.update({f"contains digit {d}": [n for n in NUMBERS if str(d) in str(n)] for d in range(1, 10)})
    return rules


RULES = list(rule_concepts().values())
N_RULES = len(RULES)
RULE_MEM = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")  # (R, 100)
# Interval grid: lo[m, n] = m, hi[m, n] = n (0-based), valid where m <= n.
_M, _N = np.meshgrid(np.arange(100), np.arange(100), indexing="ij")
VALID = _M <= _N
BETA_GRID = np.linspace(0.1, 1.5, 8)
N_GRID = len(BETA_GRID)
N_INTERVALS = int(VALID.sum())


def set_tables(examples, beta):
    """Per rule: log-lik of the set under the unrestricted rule, log mean
    lik over sub-intervals, in-concept vector for each component."""
    ex = np.array(examples) - 1
    n_ex = len(ex)
    log_full = np.zeros(N_RULES)
    log_sub = np.zeros(N_RULES)
    in_sub = np.zeros((N_RULES, 100))
    for r in range(N_RULES):
        mem = RULE_MEM[r]
        csum = np.concatenate([[0.0], np.cumsum(mem)])
        size = csum[_N + 1] - csum[_M]  # (100, 100)
        # log P(D | r ∩ [m, n])
        ll = np.zeros((100, 100))
        for x in ex:
            inside = (mem[x] > 0) & (_M <= x) & (x <= _N)
            with np.errstate(divide="ignore"):
                pin = np.where(inside & (size > 0), ALPHA / np.maximum(size, 1), 0.0)
            ll += np.log(pin + (1 - ALPHA) / 100)
        ll = np.where(VALID, beta * ll, -np.inf)
        log_full[r] = ll[0, 99]
        lse = logsumexp(ll)
        log_sub[r] = lse - np.log(N_INTERVALS)
        w = np.exp(ll - lse)  # normalised posterior over intervals given r
        # P(target t in [m, n]) = sum_{m <= t <= n} w[m, n]
        cm = np.cumsum(w, axis=0)  # sum over m' <= m
        tail = np.cumsum(cm[:, ::-1], axis=1)[:, ::-1]  # sum over n' >= n
        cover = np.array([tail[t, t] for t in range(100)])
        in_sub[r] = np.clip(cover, 0.0, 1.0) * mem
    return log_full, log_sub, in_sub


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    loglik = np.zeros((S, N_GRID, 2 * N_RULES))
    member = np.zeros((S, N_GRID, N_RULES, 100))  # in-concept under rule ∩ range
    for i, s in enumerate(unique):
        for g, b in enumerate(BETA_GRID):
            lf, ls, isub = set_tables(s, b)
            loglik[i, g] = np.concatenate([lf, ls])
            member[i, g] = isub
    return {
        "comp_log_lik": loglik,
        "range_member": member,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    comp_log_lik = pm.Data("comp_log_lik", np.zeros((1, N_GRID, 2 * N_RULES)))
    range_member = pm.Data("range_member", np.zeros((1, N_GRID, N_RULES, 100)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    q_full = pm.Beta("q_full", 2, 2)  # probability the rule is applied over all of 1..100
    # Strength of the evidence from the examples, on (0.1, 1.5); prior median 0.8.
    beta_raw = pm.Beta("beta_raw", 2, 2)
    beta = pm.Deterministic("beta", 0.1 + 1.4 * beta_raw)

    # Linear interpolation between grid points via hat weights (smooth in beta).
    pos = beta_raw * (N_GRID - 1)
    hat = pt.maximum(0.0, 1.0 - pt.abs(pos - np.arange(N_GRID, dtype="float64")))  # (G,)
    ll = (hat[None, :, None] * comp_log_lik).sum(axis=1)  # (S, 2R)
    sub_mem = (hat[None, :, None, None] * range_member).sum(axis=1)  # (S, R, 100)

    log_w = pt.log(rule_weights)
    log_prior = pt.concatenate([log_w + pt.log(q_full), log_w + pt.log1p(-q_full)])
    log_joint = log_prior[None, :] + ll
    post = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))  # (S, 2R)
    in_concept = pt.dot(post[:, :N_RULES], pt.as_tensor(RULE_MEM)) + (
        post[:, N_RULES:, None] * sub_mem
    ).sum(axis=1)  # (S, 100)
    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)
