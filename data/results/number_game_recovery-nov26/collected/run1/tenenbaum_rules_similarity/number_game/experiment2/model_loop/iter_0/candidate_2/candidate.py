"""Win-stay / lose-sample sequential concept learning (order-sensitive).

People hold a single running guess about the program (45 rules and the 5,050
intervals of 1..100, with a prior mass per rule family -- exception rules in
their own family -- and an Erlang prior over interval width) and revise it
example by example in the order shown.  After each new example they keep the
current guess with probability (L / L_max) ** stickiness, where L is how likely
that guess makes the example (size principle, 10% of examples uniform noise);
otherwise they draw a fresh guess from the posterior given all examples so far.
With stickiness = 1 this reproduces the Bayesian posterior exactly (Bonawitz et
al., 2014); below 1, guesses formed on early examples persist, so example order
matters.  "Yes" is the probability that the final guess contains the target.

For speed, intervals of one width that contain the same subset of a set's
examples are pooled: they share prior, likelihood and stay probability at every
step, so the guess spreads over them equally.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
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
    rules.update({f"contains digit {d}": [n for n in NUMBERS if str(d) in str(n)] for d in range(10)})
    return rules


def family(name):
    if name in ("odd", "even"):
        return 0
    if name.startswith("multiples"):
        return 2
    if name.startswith("ends"):
        return 3
    if name.startswith("contains"):
        return 4
    if name.startswith("powers of 2,"):
        return 5  # gerrymandered exception rules
    return 1  # primes, squares, cubes, powers


_RULES = rule_concepts()
RULES = list(_RULES.values())
N_RULES = len(RULES)
RULE_FAMILY = np.array([family(n) for n in _RULES])
FAMILY_SIZE = np.bincount(RULE_FAMILY).astype("float64")
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
EXAMPLE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
LOG_L_MAX = np.log(ALPHA + (1 - ALPHA) / 100)  # likelihood of an example under the singleton interval
INTERVAL_WIDTH = MEMBERSHIP[N_RULES:].sum(axis=1)
WIDTHS = np.arange(1, 101, dtype="float64")
N_PER_WIDTH = np.bincount(INTERVAL_WIDTH.astype("int64") - 1, minlength=100).astype("float64")

with pm.Model() as model:
    # Per set, the hypotheses are the 45 rules followed by pooled interval groups
    # (padded).  For each: the log number of intervals pooled (0 for rules, a large
    # negative number for padding), the width index (-1 for rules), the
    # log-likelihood of the example at each position (0 on padded positions) and
    # its running sum, and the mean membership of each target 1..100.
    hyp_log_count = pm.Data("hyp_log_count", np.zeros((1, 1)))
    hyp_width = pm.Data("hyp_width", np.zeros((1, 1), dtype="int64"))
    step_log_lik = pm.Data("step_log_lik", np.zeros((1, MAX_EX, 1)))
    cum_log_lik = pm.Data("cum_log_lik", np.zeros((1, MAX_EX, 1)))
    step_mask = pm.Data("step_mask", np.zeros((1, MAX_EX)))  # 1 where a real example is shown
    hyp_membership = pm.Data("hyp_membership", np.zeros((1, 1, 100)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    family_weight = pm.Dirichlet("family_weight", a=np.ones(7))
    log_w = pt.log(family_weight)
    interval_scale = pm.LogNormal("interval_scale", mu=np.log(20.0), sigma=1.0)
    # Stickiness of the running guess: 1 = Bayesian, < 1 = early guesses persist.
    stickiness = pm.LogNormal("stickiness", mu=0.0, sigma=1.0)

    erlang = np.log(WIDTHS) - WIDTHS / interval_scale
    log_per_width = erlang - pt.logsumexp(erlang + np.log(N_PER_WIDTH))
    rule_prior = (log_w[:6] - np.log(FAMILY_SIZE))[RULE_FAMILY]
    interval_prior = log_w[6] + log_per_width[pt.maximum(hyp_width[:, N_RULES:], 0)]
    log_prior = pt.concatenate([pt.broadcast_to(rule_prior[None, :], (hyp_width.shape[0], N_RULES)), interval_prior], axis=1)
    log_prior = log_prior + hyp_log_count

    def posterior(k):
        lj = log_prior + cum_log_lik[:, k, :]
        return pt.exp(lj - pt.logsumexp(lj, axis=1, keepdims=True))

    guess = posterior(0)
    for k in range(1, MAX_EX):
        m = step_mask[:, k][:, None]
        stay = pt.exp(stickiness * (step_log_lik[:, k, :] - LOG_L_MAX) * m)  # 1 on padded steps
        kept = guess * stay
        resample = pt.clip(1.0 - kept.sum(axis=1, keepdims=True), 0.0, 1.0)
        guess = kept + resample * posterior(k)

    in_concept = pt.batched_dot(guess, hyp_membership)
    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


INTERVAL_BOUNDS = [(m, n) for m in NUMBERS for n in range(m, 101)]


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    groups = []
    for s in unique:
        g = {}
        for j, (a, b) in enumerate(INTERVAL_BOUNDS):
            g.setdefault((b - a + 1, tuple(a <= x <= b for x in s)), []).append(N_RULES + j)
        groups.append(list(g.values()))
    n_h = N_RULES + max(len(g) for g in groups)
    log_count = np.full((len(unique), n_h), -1e4)
    width = np.full((len(unique), n_h), -1, dtype="int64")
    step = np.zeros((len(unique), MAX_EX, n_h))
    mask = np.zeros((len(unique), MAX_EX))
    member = np.zeros((len(unique), n_h, 100))
    for i, (s, g) in enumerate(zip(unique, groups)):
        idx = list(range(N_RULES)) + [grp[0] for grp in g]
        n = len(idx)
        log_count[i, :N_RULES] = 0.0
        log_count[i, N_RULES:n] = np.log([len(grp) for grp in g])
        width[i, N_RULES:n] = [int(INTERVAL_WIDTH[grp[0] - N_RULES]) - 1 for grp in g]
        member[i, :N_RULES] = MEMBERSHIP[:N_RULES]
        member[i, N_RULES:n] = [MEMBERSHIP[grp].mean(axis=0) for grp in g]
        for k, x in enumerate(s[:MAX_EX]):
            step[i, k, :n] = EXAMPLE_LOG_LIK[idx, x - 1]
            mask[i, k] = 1.0
    return {
        "hyp_log_count": log_count,
        "hyp_width": width,
        "step_log_lik": step,
        "cum_log_lik": np.cumsum(step, axis=1),
        "step_mask": mask,
        "hyp_membership": member,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
