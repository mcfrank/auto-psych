"""Memory-limited Bayesian concept learning.

People do Bayesian concept learning (Bigelow & Piantadosi mixture: 45 equally
likely rules with total prior lambda_0, every interval of 1..100 with prior
(1 - lambda_0) / 100^2, examples drawn from the concept with probability 0.9
- the size principle - else uniformly), but under a working-memory limit:
each example is retained independently with probability r and the learner
reasons as if the forgotten ones were never seen. "Yes" is the posterior
probability that the concept contains the target, averaged over which subset
of examples was remembered. The single distortion is the retention rate r.
"""
import itertools

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
NUMBERS = np.arange(1, 101)


def _is_prime(n):
    return n > 1 and all(n % a for a in range(2, int(n ** 0.5) + 1))


def rule_concepts():
    rules = [
        [n for n in NUMBERS if n % 2 == 1],
        [n for n in NUMBERS if _is_prime(n)],
        [n * n for n in range(1, 11)],
        [n ** 3 for n in range(1, 5)],
        [2, 4, 8, 16, 64],
        [2, 4, 8, 16, 32, 37, 64],
        list(range(2, 101, 2)),
    ]
    rules += [list(range(k, 101, k)) for k in [3, 4, 5, 6, 7, 8, 9, 11, 12]]
    rules += [[k ** j for j in range(1, 8) if k ** j <= 100] for k in range(2, 11)]
    rules += [[n for n in NUMBERS if n % 10 == d] for d in range(10)]
    rules += [[n for n in NUMBERS if str(d) in str(n)] for d in range(10)]
    return rules


RULES = rule_concepts()
N_RULES = len(RULES)
INTERVALS = [list(range(a, b + 1)) for a in NUMBERS for b in range(a, 101)]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
N_HYP = MEMBERSHIP.shape[0]
EXAMPLE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
IS_INTERVAL = np.r_[np.zeros(N_RULES), np.ones(N_HYP - N_RULES)]


def prepare_observed(rows):
    """Per distinct set, enumerate every remembered subset of its examples."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    subsets = {}  # distinct remembered subset -> column
    kept, forgot, cells = [], [], []
    for i, s in enumerate(unique):
        for k in range(len(s) + 1):
            for c in itertools.combinations(s, k):
                key = tuple(sorted(c))
                if key not in subsets:
                    subsets[key] = len(subsets)
                cells.append((i, subsets[key]))
                kept.append(k)
                forgot.append(len(s) - k)
    # A set may reach the same distinct subset in several ways (repeated examples),
    # so the weights stay per (set, subset-draw) and are summed by the assign matrix.
    n_draws = len(cells)
    assign = np.zeros((len(unique), n_draws))
    pick = np.zeros((n_draws, len(subsets)))
    for j, (i, u) in enumerate(cells):
        assign[i, j] = 1.0
        pick[j, u] = 1.0
    sub_ll = [
        EXAMPLE_LOG_LIK[:, np.array(key, dtype=int) - 1].sum(axis=1) if key else np.zeros(N_HYP)
        for key in subsets
    ]
    return {
        "subset_log_lik": np.stack(sub_ll),
        "subset_kept": np.array(kept, dtype="float64"),
        "subset_forgot": np.array(forgot, dtype="float64"),
        "subset_assign": assign,
        "draw_subset": pick,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    subset_log_lik = pm.Data("subset_log_lik", np.zeros((1, N_HYP)))
    subset_kept = pm.Data("subset_kept", np.zeros(1, dtype="float64"))
    subset_forgot = pm.Data("subset_forgot", np.zeros(1, dtype="float64"))
    subset_assign = pm.Data("subset_assign", np.ones((1, 1)))  # (sets, draws)
    draw_subset = pm.Data("draw_subset", np.ones((1, 1)))  # (draws, distinct subsets)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    retain = pm.Beta("retain", 2, 2)  # probability each example is remembered

    is_int = pt.as_tensor(IS_INTERVAL)
    log_prior = (1 - is_int) * (pt.log(lambda_0) - np.log(N_RULES)) + is_int * (
        pt.log1p(-lambda_0) - 2 * np.log(100)
    )
    posterior = pt.special.softmax(subset_log_lik + log_prior[None, :], axis=1)
    pred_subset = posterior @ pt.as_tensor(MEMBERSHIP)  # (distinct subsets, 100)
    w = pt.exp(subset_kept * pt.log(retain) + subset_forgot * pt.log1p(-retain))
    pred_set = ((subset_assign * w[None, :]) @ draw_subset) @ pred_subset  # (sets, 100)

    p_yes = pm.Deterministic("p_yes", pt.clip(pred_set.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)
