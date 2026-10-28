"""Periodic-spacing Bayesian concept learning.

People learn the hidden program's concept by Bayesian inference with the size
principle (10% of examples uniform noise), over the 42 named rules of
bigelow_piantadosi_independent (each with its own Dirichlet prior weight), all
intervals of 1-100 (equally likely), and -- the new ingredient -- every
arithmetic progression "every d-th number starting from r" (step d = 2..50,
offset r = 1..d), so that a target continuing the examples' common spacing is
generalized to even when no named rule covers it. A progression's prior falls
off with its step as d^-gamma (shorter steps more plausible), shared equally
over its d offsets. "Yes" is the posterior probability the concept contains
the target.

For speed, hypotheses whose prior weights are tied (all intervals; all offsets
of one step) are aggregated per example set into one hypothesis carrying their
summed likelihood and the share of their posterior containing each target;
the posterior is exactly the one over every individual hypothesis.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

EPSILON = 0.1
NUMBERS = np.arange(1, 101)
STEPS = np.arange(2, 51)


def rule_sets():
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
    return rules


def membership(sets):
    return np.array([np.isin(NUMBERS, c) for c in sets], dtype="float64")


RULE_MEM = membership(rule_sets())
N_RULES = RULE_MEM.shape[0]
INTERVAL_MEM = membership([range(m, n + 1) for m in range(1, 101) for n in range(m, 101)])
PROG_MEM = [membership([range(r, 101, d) for r in range(1, d + 1)]) for d in STEPS]


def log_lik(mem, examples):
    """log P(examples | h) under strong sampling with noise, per hypothesis row."""
    p = mem * (1 - EPSILON) / mem.sum(axis=1, keepdims=True) + EPSILON / 100
    return np.log(p[:, [x - 1 for x in examples]]).sum(axis=1)


def aggregate(mem, examples):
    """Mean likelihood over equally weighted hypotheses, and posterior share containing each number."""
    ll = log_lik(mem, examples)
    top = ll.max()
    w = np.exp(ll - top)
    return top + np.log(w.mean()), (w @ mem) / w.sum()


with pm.Model() as model:
    set_log_lik = pm.Data("set_log_lik", np.zeros((1, 1)))  # (sets, rules + steps + 1)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_member = pm.Data("cell_member", np.zeros((1, 1)))  # (cells, hyp): share containing target
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    mix = pm.Dirichlet("mix", a=np.array([2.0, 2.0, 2.0]))  # rules, progressions, intervals
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    gamma = pm.HalfNormal("gamma", sigma=2.0)  # step penalty

    step_logits = -gamma * np.log(STEPS)
    log_steps = step_logits - pt.logsumexp(step_logits)
    log_prior = pt.concatenate([
        pt.log(mix[0]) + pt.log(rule_weights),
        pt.log(mix[1]) + log_steps,
        pt.log(mix[2])[None],
    ])
    log_joint = log_prior[None, :] + set_log_lik
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = (posterior[cell_set] * cell_member).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    n_hyp = N_RULES + len(STEPS) + 1
    set_ll = np.zeros((len(unique), n_hyp))
    set_share = np.zeros((len(unique), n_hyp, 100))
    for i, s in enumerate(unique):
        set_ll[i, :N_RULES] = log_lik(RULE_MEM, s)
        set_share[i, :N_RULES] = RULE_MEM
        for k, mem in enumerate(PROG_MEM):
            set_ll[i, N_RULES + k], set_share[i, N_RULES + k] = aggregate(mem, s)
        set_ll[i, -1], set_share[i, -1] = aggregate(INTERVAL_MEM, s)
    cells = list(dict.fromkeys(zip(sets, targets)))
    cell_index = {c: i for i, c in enumerate(cells)}
    return {
        "set_log_lik": set_ll,
        "cell_set": np.array([index[s] for s, _ in cells], dtype="int64"),
        "cell_member": np.stack([set_share[index[s], :, t - 1] for s, t in cells]),
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
