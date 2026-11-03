"""Win-stay, lose-shift concept tracking with rule-specific prior weights.

Refinement of first_guess_win_stay_concepts. The one change: each rule has its
own fitted prior weight (a Dirichlet over rules, "ends in 0" / "contains digit
0" -- the same set -- sharing one component split by a Beta, as in
erlang_interval_prior_concepts), instead of rules sharing their mass equally.
Everything below is otherwise the incumbent's mechanism.

People hold a single working guess about the concept (one of 45 rules or an
interval of 1..100). After the first example the guess is drawn from the
Bayesian posterior (size principle, 10% noisy examples, rules sharing their prior
mass equally, an Erlang prior on interval size). As each further example arrives,
if it lies in the current guess the learner keeps it with probability
`stay` (no size-principle re-weighting); otherwise (contradiction, or not
sticking) the guess is redrawn from the posterior given all examples so far.
"Yes" is the probability that the final guess contains the target. With
stay = 0 this is the full posterior learner; stay > 0 makes early examples
weigh more (a primacy / anchoring effect).

For speed, intervals are grouped per example set by (size, which examples they
contain): members of a group share prior, likelihood at every step and
containment of every example, so the guess-tracking dynamics are exact on the
groups (each carrying its log count and mean membership of each target).
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
NUMBERS = np.arange(1, 101)
MAX_N = 4
G_MAX = 480  # interval groups per set (padded)
EMPTY_LOG_COUNT = -1e4


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
RULE_EX_LOG_LIK = np.log(ALPHA * RULE_MEMBERSHIP / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
LO = np.array([m for m in NUMBERS for n in range(m, 101)])
HI = np.array([n for m in NUMBERS for n in range(m, 101)])
INTERVAL_SIZE = (HI - LO + 1).astype("float64")
INTERVAL_MEMBERSHIP = ((NUMBERS[None, :] >= LO[:, None]) & (NUMBERS[None, :] <= HI[:, None])).astype("float64")
N_HYP = N_RULES + G_MAX
END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

with pm.Model() as model:
    prefix_log_lik = pm.Data("prefix_log_lik", np.zeros((1, MAX_N, N_HYP)))  # log lik of examples 1..k
    contains = pm.Data("contains", np.ones((1, MAX_N, N_HYP)))  # example k lies in hypothesis
    step_mask = pm.Data("step_mask", np.zeros((1, MAX_N)))  # example k exists
    group_log_count = pm.Data("group_log_count", np.zeros((1, G_MAX)))
    group_size = pm.Data("group_size", np.ones((1, G_MAX)))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # set of each asked (set, target) cell
    cell_member = pm.Data("cell_member", np.zeros((1, N_HYP)))  # share of hypothesis containing the target
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # cell of each trial
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Rules get prior mass lambda_0, split across rules by fitted weights.
    lambda_0 = pm.Beta("lambda_0", 2, 2)
    set_weights = pm.Dirichlet("set_weights", a=SET_CONCENTRATION)
    zero_split = pm.Beta("zero_split", 2, 2)
    shared = set_weights[END_0]
    rule_weights = pm.Deterministic("rule_weights", pt.concatenate([
        set_weights[:END_0], (zero_split * shared)[None], set_weights[END_0 + 1:CONTAINS_0],
        ((1 - zero_split) * shared)[None], set_weights[CONTAINS_0:],
    ]))
    log_sigma = pm.Normal("log_sigma", mu=np.log(10.0), sigma=1.0)
    sigma = pt.exp(log_sigma)
    size = pt.as_tensor(INTERVAL_SIZE)
    log_norm = pt.logsumexp(pt.log(size) - size / sigma)
    group_log_prior = pt.log(group_size) - group_size / sigma - log_norm + group_log_count
    log_prior = pt.concatenate([
        pt.zeros((group_size.shape[0], N_RULES)) + (pt.log(lambda_0) + pt.log(rule_weights))[None, :],
        pt.log1p(-lambda_0) + group_log_prior,
    ], axis=1)  # (sets, hyp)

    # Probability of keeping a guess that the new example fits.
    stay = pm.Beta("stay", 2, 2)

    log_joint = log_prior[:, None, :] + prefix_log_lik
    post = pt.exp(log_joint - pt.logsumexp(log_joint, axis=2, keepdims=True))  # (sets, steps, hyp)

    q = post[:, 0, :]
    for k in range(1, MAX_N):
        kept = stay * q * contains[:, k, :]
        updated = kept + (1 - kept.sum(axis=1, keepdims=True)) * post[:, k, :]
        m = step_mask[:, k][:, None]
        q = m * updated + (1 - m) * q

    in_cell = (q[cell_set] * cell_member).sum(axis=1)
    p_yes = pm.Deterministic("p_yes", pt.clip(in_cell[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    n_sets = len(unique)
    prefix = np.zeros((n_sets, MAX_N, N_HYP))
    contains = np.ones((n_sets, MAX_N, N_HYP))
    mask = np.zeros((n_sets, MAX_N))
    log_count = np.full((n_sets, G_MAX), EMPTY_LOG_COUNT)
    gsize = np.ones((n_sets, G_MAX))
    in_concept = np.zeros((n_sets, G_MAX, 100))
    for i, s in enumerate(unique):
        inside = np.stack([(LO <= x) & (HI >= x) for x in s], axis=1)  # (intervals, examples)
        code = (inside * (1 << np.arange(len(s)))).sum(axis=1)
        keys, group = np.unique(np.stack([INTERVAL_SIZE.astype(int), code], axis=1), axis=0, return_inverse=True)
        group = group.reshape(-1)
        n_g = len(keys)
        assert n_g <= G_MAX
        count = np.bincount(group, minlength=n_g).astype("float64")
        members = np.zeros((n_g, 100))
        np.add.at(members, group, INTERVAL_MEMBERSHIP)
        log_count[i, :n_g] = np.log(count)
        gsize[i, :n_g] = keys[:, 0]
        in_concept[i, :n_g] = members / count[:, None]
        g_in = np.zeros((n_g, len(s)))
        for j in range(len(s)):
            g_in[:, j] = (keys[:, 1] >> j) & 1
        g_ex_ll = np.where(g_in > 0, np.log(ALPHA / keys[:, :1] + (1 - ALPHA) / 100), np.log((1 - ALPHA) / 100))
        running = np.zeros(N_HYP)
        for k in range(MAX_N):
            if k < len(s):
                step_ll = np.zeros(N_HYP)
                step_ll[:N_RULES] = RULE_EX_LOG_LIK[:, s[k] - 1]
                step_ll[N_RULES:N_RULES + n_g] = g_ex_ll[:, k]
                running = running + step_ll
                contains[i, k, :N_RULES] = RULE_MEMBERSHIP[:, s[k] - 1]
                contains[i, k, N_RULES:] = 0.0
                contains[i, k, N_RULES:N_RULES + n_g] = g_in[:, k]
                mask[i, k] = 1.0
            prefix[i, k] = running
    cells = list(dict.fromkeys((index[s], int(r["target"])) for s, r in zip(sets, rows)))
    cell_index = {c: j for j, c in enumerate(cells)}
    cell_member = np.zeros((len(cells), N_HYP))
    for j, (i, t) in enumerate(cells):
        cell_member[j, :N_RULES] = RULE_MEMBERSHIP[:, t - 1]
        cell_member[j, N_RULES:] = in_concept[i, :, t - 1]
    return {
        "prefix_log_lik": prefix,
        "contains": contains,
        "step_mask": mask,
        "group_log_count": log_count,
        "group_size": gsize,
        "cell_set": np.array([i for i, _ in cells], dtype="int64"),
        "cell_member": cell_member,
        "row_cell": np.array([cell_index[(index[s], int(r["target"]))] for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
