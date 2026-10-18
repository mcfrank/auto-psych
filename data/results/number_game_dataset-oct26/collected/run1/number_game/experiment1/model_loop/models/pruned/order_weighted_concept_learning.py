"""Order-weighted Bayesian concept learning (primacy / recency in evidence).

People do Bayesian concept learning over mathematical rules and numerical
intervals, with the size principle, but they do not weigh the examples
equally: how much each example shifts their belief depends on where it came
in the order shown, growing or shrinking steadily from first to last (primacy
or recency), while the total evidence stays the same. So the same numbers
shown in a different order lead to different generalizations.

Hypothesis space, prior and likelihood follow the Bigelow & Piantadosi
learner (42 rules with inferred prior weights, a share p_mix for the rules,
5,050 equally likely intervals, 10% of examples uniform noise). The one change:
the log-likelihood of the k-th of n examples is multiplied by
w_k = n * rho**(k-1) / sum_j rho**(j-1), so the weights sum to n (same total
evidence) and rho > 1 is recency, rho < 1 primacy, rho = 1 the equal-weight
learner. Because the weights depend on rho the intervals no longer collapse
into one hypothesis; intervals sharing a length and the examples they contain
are grouped (exact), and the in-concept probability is computed only for the
(set, target) cells that were asked about.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

EPSILON = 0.1
NUMBERS = np.arange(1, 101)
MAX_N = 4


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
N_RULES, N_INTERVALS = len(RULES), len(INTERVALS)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_EXAMPLE_LOG_LIK = np.log(
    RULE_MEMBERSHIP * (1 - EPSILON) / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + EPSILON / 100
)
G_MAX = 600  # interval groups per set (at most 473 on this design)


def interval_groups(examples):
    """Intervals with the same length and the same examples inside share a likelihood
    under any weighting; group them: (count, per-position log-lik, coverage of 1..100)."""
    groups = {}
    for m in range(1, 101):
        for n in range(m, 101):
            key = (n - m + 1, tuple(m <= x <= n for x in examples))
            g = groups.setdefault(key, [0, np.zeros(100)])
            g[0] += 1
            g[1][m - 1:n] += 1
    out = []
    for (length, inside), (count, cover) in groups.items():
        ll = [np.log((1 - EPSILON) / length + EPSILON / 100) if i else np.log(EPSILON / 100) for i in inside]
        out.append((count, ll, cover / count))
    return out


def prepare_observed(rows):
    """Per distinct set: per-position log-likelihoods; per distinct cell: membership."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    set_index = {s: i for i, s in enumerate(unique)}
    n_hyp = N_RULES + G_MAX
    pos_ll = np.zeros((len(unique), MAX_N, n_hyp))
    log_count = np.full((len(unique), G_MAX), -1e4)  # padding groups get no prior mass
    member = np.zeros((len(unique), n_hyp, 100))
    n_ex = np.zeros(len(unique))
    for i, s in enumerate(unique):
        n_ex[i] = len(s)
        member[i, :N_RULES] = RULE_MEMBERSHIP
        for k, x in enumerate(s):
            pos_ll[i, k, :N_RULES] = RULE_EXAMPLE_LOG_LIK[:, x - 1]
        for g, (count, ll, cover) in enumerate(interval_groups(s)):
            log_count[i, g] = np.log(count)
            pos_ll[i, :len(s), N_RULES + g] = ll
            member[i, N_RULES + g] = cover
    cells = [(set_index[s], int(r["target"])) for s, r in zip(sets, rows)]
    unique_cells = list(dict.fromkeys(cells))
    cell_index = {c: i for i, c in enumerate(unique_cells)}
    return {
        "pos_log_lik": pos_ll,
        "interval_log_count": log_count,
        "n_examples": n_ex,
        "cell_set": np.array([c[0] for c in unique_cells], dtype="int64"),
        "cell_member": np.stack([member[s, :, t - 1] for s, t in unique_cells]),
        "row_cell": np.array([cell_index[c] for c in cells], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    pos_log_lik = pm.Data("pos_log_lik", np.zeros((1, MAX_N, N_RULES + G_MAX)))  # (sets, position, hyps)
    interval_log_count = pm.Data("interval_log_count", np.zeros((1, G_MAX)))  # (sets, groups)
    n_examples = pm.Data("n_examples", np.ones(1))  # (sets,)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # (cells,)
    cell_member = pm.Data("cell_member", np.zeros((1, N_RULES + G_MAX)))  # (cells, hyps)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # (trials,)
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 4, 4)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    # Log ratio of successive examples' weights: > 0 recency, < 0 primacy.
    log_rho = pm.Normal("log_rho", mu=0.0, sigma=1.0)

    position = pt.arange(MAX_N, dtype="float64")
    shown = pt.lt(position[None, :], n_examples[:, None])  # (sets, position)
    raw = pt.switch(shown, pt.exp(log_rho * (position[None, :] - (n_examples[:, None] - 1) / 2)), 0.0)
    weights = n_examples[:, None] * raw / raw.sum(axis=1, keepdims=True)
    log_lik = (weights[:, :, None] * pos_log_lik).sum(axis=1)  # (sets, hyps)

    log_prior_rules = pt.log(p_mix) + pt.log(rule_weights)
    log_prior_intervals = pt.log1p(-p_mix) - np.log(N_INTERVALS) + interval_log_count
    log_joint = pt.concatenate(
        [pt.broadcast_to(log_prior_rules[None, :], (log_lik.shape[0], N_RULES)), log_prior_intervals], axis=1
    ) + log_lik
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = (posterior[cell_set] * cell_member).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)
