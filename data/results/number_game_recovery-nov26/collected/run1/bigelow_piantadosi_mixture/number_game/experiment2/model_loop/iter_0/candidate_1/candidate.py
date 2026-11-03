"""Independent single-example generalization.

People never pool the examples into a single inferred concept. Each example on
its own triggers Bayesian generalization (over 45 mathematical rules and all
numerical ranges of 1-100, size principle, 10% noisy examples, rules sharing a
fitted prior share equally, ranges weighted by an Erlang density on their size),
and the target is endorsed according to the average of these separate
single-example generalizations. Adding examples therefore neither sharpens
generalization onto a shared rule nor fills the gaps between them as a range.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
NUMBERS = np.arange(1, 101)
MAX_N = 4


def _rules():
    rules = [
        [n for n in NUMBERS if n % 2 == 1],
        [n for n in NUMBERS if n > 1 and all(n % a for a in range(2, int(n**0.5) + 1))],
        [n * n for n in range(1, 11)],
        [n**3 for n in range(1, 5)],
        [2, 4, 8, 16, 64],
        [2, 4, 8, 16, 32, 37, 64],
        list(range(2, 101, 2)),
    ]
    rules += [list(range(m, 101, m)) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]]
    rules += [[m**k for k in range(1, 8) if m**k <= 100] for m in range(2, 11)]
    rules += [[n for n in NUMBERS if n % 10 == d] for d in range(10)]
    rules += [[n for n in NUMBERS if str(d) in str(n)] for d in range(10)]
    return rules


RULES = _rules()
N_RULES = len(RULES)
RULE_MEM = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
LO = np.array([m for m in NUMBERS for n in range(m, 101)])
HI = np.array([n for m in NUMBERS for n in range(m, 101)])
INT_SIZE = (HI - LO + 1).astype("float64")
INT_MEM = ((NUMBERS[None, :] >= LO[:, None]) & (NUMBERS[None, :] <= HI[:, None])).astype("float64")
MEM = np.concatenate([RULE_MEM, INT_MEM], axis=0)  # (hyp, 100)
HYP_SIZE = MEM.sum(axis=1)
# Log-likelihood of one example x under each hypothesis: (100 examples, hyp)
EX_LL = np.log(ALPHA * MEM.T / HYP_SIZE[None, :] + (1 - ALPHA) / 100)


with pm.Model() as model:
    ex_idx = pm.Data("ex_idx", np.zeros((1, MAX_N), dtype="int64"))  # example - 1
    ex_w = pm.Data("ex_w", np.ones((1, MAX_N)))  # 1/n for real examples, 0 for padding
    target_idx = pm.Data("target_idx", np.zeros(1, dtype="int64"))  # target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    log_sigma = pm.Normal("log_sigma", mu=np.log(10.0), sigma=1.0)
    sigma = pt.exp(log_sigma)

    size = pt.as_tensor(INT_SIZE)
    int_lp = pt.log(size) - size / sigma
    int_lp = int_lp - pt.logsumexp(int_lp) + pt.log1p(-lambda_0)
    rule_lp = pt.zeros(N_RULES) + pt.log(lambda_0) - np.log(N_RULES)
    log_prior = pt.concatenate([rule_lp, int_lp])

    # Generalization from each single example x to each target t: (100, 100)
    post = pt.special.softmax(log_prior[None, :] + pt.as_tensor(EX_LL), axis=1)
    gen = pt.dot(post, pt.as_tensor(MEM))

    per_ex = gen[ex_idx, target_idx[:, None]]  # (rows, MAX_N)
    p = (per_ex * ex_w).sum(axis=1)
    p_yes = pm.Deterministic("p_yes", pt.clip(p, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    n = len(rows)
    idx = np.zeros((n, MAX_N), dtype="int64")
    w = np.zeros((n, MAX_N))
    for i, r in enumerate(rows):
        ex = [int(x) for x in str(r["number_set"]).split(",")]
        idx[i, :len(ex)] = np.array(ex) - 1
        w[i, :len(ex)] = 1.0 / len(ex)
    return {
        "ex_idx": idx,
        "ex_w": w,
        "target_idx": np.array([int(r["target"]) - 1 for r in rows], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
