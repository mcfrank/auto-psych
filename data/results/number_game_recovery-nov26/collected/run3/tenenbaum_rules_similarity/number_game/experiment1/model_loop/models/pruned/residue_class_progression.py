"""Residue-class (arithmetic progression) concept learning.

People assume the hidden program produces an arithmetic progression: all
numbers in 1-100 that share a remainder modulo some step size (step 1 = every
number, 2 = odds/evens, 10 = a shared final digit, 12 = every twelfth number,
...). They consider every step whose residue class contains all the examples,
favour small steps a priori but tighter (larger-step) classes when several
examples fit them (the size principle), and answer "yes" with the posterior
probability that the target lies in the same progression as the examples.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_STEP = 50
STEPS = np.arange(1, MAX_STEP + 1)
LOG_STEP = np.log(STEPS).astype("float64")


def prepare_observed(rows):
    n = len(rows)
    consistent = np.zeros((n, MAX_STEP))
    member = np.zeros((n, MAX_STEP))
    n_log_size = np.zeros((n, MAX_STEP))
    rating = np.zeros(n, dtype="int64")
    cache = {}
    for i, r in enumerate(rows):
        key = r["number_set"]
        if key not in cache:
            ex = [int(x) for x in key.split(",")]
            cons = np.array([all((e - ex[0]) % m == 0 for e in ex) for m in STEPS], float)
            sizes = np.array([sum(1 for v in range(1, 101) if (v - ex[0]) % m == 0) for m in STEPS], float)
            cache[key] = (ex, cons, len(ex) * np.log(sizes))
        ex, cons, nls = cache[key]
        t = int(r["target"])
        consistent[i] = cons
        member[i] = [(t - ex[0]) % m == 0 for m in STEPS]
        n_log_size[i] = nls
        rating[i] = int(float(r["rating"]))
    return {"consistent": consistent, "member": member,
            "n_log_size": n_log_size, "rating": rating}


with pm.Model() as model:
    consistent = pm.Data("consistent", np.ones((1, MAX_STEP)))
    member = pm.Data("member", np.ones((1, MAX_STEP)))
    n_log_size = pm.Data("n_log_size", np.zeros((1, MAX_STEP)))

    # Prior preference for small steps (power law over step size).
    alpha = pm.HalfNormal("alpha", sigma=2.0)
    # Strength of the size principle (1 = strong sampling).
    gamma = pm.HalfNormal("gamma", sigma=1.0)
    # Lapse: with this probability the answer is a guess at base rate.
    lapse = pm.Beta("lapse", alpha=2.0, beta=8.0)
    guess = pm.Beta("guess", alpha=2.0, beta=2.0)

    log_w = -alpha * LOG_STEP[None, :] - gamma * n_log_size + (consistent - 1.0) * 1e3
    w = pt.special.softmax(log_w, axis=1)
    p_in = pt.sum(w * member, axis=1)
    p_yes = pm.Deterministic(
        "p_yes", pt.clip((1 - lapse) * p_in + lapse * guess, 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
