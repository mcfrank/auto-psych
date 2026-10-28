"""Win-stay, lose-shift concept tracking.

People hold a single running guess about the concept and revise it example by
example in the order shown. After the first example they adopt a concept in
proportion to its Bayesian posterior (42 weighted rules and all intervals of
1..100, size principle with 10% noisy examples, evidence tempered as in the
incumbent); they keep it while each new example lies in it, and when an example
contradicts it they switch to a concept drawn from the posterior given all
examples so far. "Yes" is the probability that the final guess contains the
target, with occasional guessing as in the incumbent. Single change from
tempered_evidence_concept_guessing: the full posterior is replaced by the
distribution of this sequential win-stay/lose-shift guess (order-dependent).
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

EPSILON = 0.1
NUMBERS = np.arange(1, 101)
MAX_SET = 4


def concept_sets():
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
EXAMPLE_LOG_LIK = np.log(MEMBERSHIP * (1 - EPSILON) / MEMBERSHIP.sum(axis=1, keepdims=True) + EPSILON / 100)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]
INTERVAL_WIDTH = INTERVAL_MEMBERSHIP.sum(axis=1).astype("int64")
LOG_N_INTERVALS = np.log(len(INTERVALS))


def _set_groups(s):
    """Group a set's intervals by (width, which examples by position they contain):
    members of a group share prior, likelihood at every step and consistency."""
    ex = np.array(s) - 1
    pattern = INTERVAL_MEMBERSHIP[:, ex] @ (2 ** np.arange(len(s)))
    keys = INTERVAL_WIDTH * 64 + pattern.astype("int64")
    uniq, inv = np.unique(keys, return_inverse=True)
    return ex, uniq, inv


def prepare_observed(rows):
    """Rows carry `number_set` (in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    grouped = [_set_groups(s) for s in unique]
    n_groups = max(len(g[1]) for g in grouped)
    n_sets = len(unique)
    n_hyp = N_RULES + n_groups
    # Per set and step k: membership of example k (consistency) and cumulative log-lik.
    consistent = np.ones((n_sets, MAX_SET, n_hyp))
    cum_ll = np.zeros((n_sets, MAX_SET, n_hyp))
    log_count = np.full((n_sets, n_hyp), -700.0)
    log_count[:, :N_RULES] = 0.0
    share = []  # per set: (groups, 100)
    for i, (s, (ex, uniq, inv)) in enumerate(zip(unique, grouped)):
        g_n = len(uniq)
        rep = np.array([np.flatnonzero(inv == g)[0] for g in range(g_n)])
        counts = np.bincount(inv, minlength=g_n)
        log_count[i, N_RULES:N_RULES + g_n] = np.log(counts)
        sh = np.zeros((n_groups, 100))
        for g in range(g_n):
            sh[g] = INTERVAL_MEMBERSHIP[inv == g].mean(axis=0)
        share.append(sh)
        run = np.zeros(n_hyp)
        for k in range(MAX_SET):
            if k < len(ex):
                ll = np.zeros(n_hyp)
                ll[:N_RULES] = EXAMPLE_LOG_LIK[:N_RULES, ex[k]]
                ll[N_RULES:N_RULES + g_n] = EXAMPLE_LOG_LIK[N_RULES + rep, ex[k]]
                run = run + ll
                consistent[i, k, :N_RULES] = RULE_MEMBERSHIP[:, ex[k]]
                consistent[i, k, N_RULES:N_RULES + g_n] = INTERVAL_MEMBERSHIP[rep, ex[k]]
                consistent[i, k, N_RULES + g_n:] = 1.0
            cum_ll[i, k] = run  # padded steps: all consistent, nobody shifts
    share = np.stack(share)
    set_idx = np.array([index[s] for s in sets], dtype="int64")
    tgt = np.array([int(r["target"]) - 1 for r in rows], dtype="int64")
    return {
        "consistent": consistent,
        "cum_ll": cum_ll,
        "hyp_log_count": log_count,
        "group_in_concept": share[set_idx, :, tgt],
        "row_set": set_idx,
        "row_cell": set_idx * 100 + tgt,
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    consistent = pm.Data("consistent", np.ones((1, MAX_SET, N_RULES + 1)))  # (sets, steps, hyps)
    cum_ll = pm.Data("cum_ll", np.zeros((1, MAX_SET, N_RULES + 1)))  # log P(x_1..x_k | h)
    hyp_log_count = pm.Data("hyp_log_count", np.zeros((1, N_RULES + 1)))  # (sets, hyps)
    group_in_concept = pm.Data("group_in_concept", np.zeros((1, 1)))  # (trials, groups)
    row_set = pm.Data("row_set", np.zeros(1, dtype="int64"))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 4, 4)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    evidence_weight = pm.LogNormal("evidence_weight", mu=0.0, sigma=0.5)

    log_prior = pt.concatenate(
        [
            pt.broadcast_to(pt.log(p_mix) + pt.log(rule_weights)[None, :], (hyp_log_count.shape[0], N_RULES)),
            pt.log1p(-p_mix) - LOG_N_INTERVALS + hyp_log_count[:, N_RULES:],
        ],
        axis=1,
    )

    def posterior(k):
        lj = log_prior + evidence_weight * cum_ll[:, k, :]
        return pt.exp(lj - pt.logsumexp(lj, axis=1, keepdims=True))

    # The running guess: adopt from the posterior after example 1; on each later
    # example keep the guess if it fits, else redraw from the current posterior.
    guess = posterior(0)
    for k in range(1, MAX_SET):
        kept = guess * consistent[:, k, :]
        guess = kept + (1.0 - kept.sum(axis=1, keepdims=True)) * posterior(k)

    rule_part = (guess[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP)).reshape((-1,))[row_cell]
    interval_part = (guess[:, N_RULES:][row_set] * group_in_concept).sum(axis=1)
    concept_p = pt.clip(rule_part + interval_part, 0.0, 1.0)

    guess_rate = pm.Beta("guess_rate", 2, 8)
    guess_bias = pm.Beta("guess_bias", 2, 2)
    p_yes = pm.Deterministic(
        "p_yes", pt.clip((1 - guess_rate) * concept_p + guess_rate * guess_bias, 1e-6, 1 - 1e-6)
    )
    pm.Bernoulli("response", p=p_yes, observed=rating)
