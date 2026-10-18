"""Bayesian concept learning over feature-based number concepts.

People generalize as Bayesian concept learners over a small space of concepts
built only from the features they attend to: "ends in digit d", "even"/"odd",
"multiples of k", a contiguous band of decades, or "any number". Each concept
consistent with all examples is weighted by its family's prior plausibility and
by the size principle (with an inferred strength), and "yes" follows the
posterior belief that the target is in the program's concept (read out on a
log-odds scale with an inferred bias and slope).
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMS = np.arange(1, 101)


def _hypotheses():
    hyps, fams = [], []
    for d in range(10):
        hyps.append(NUMS % 10 == d)
        fams.append(0)
    hyps.append(NUMS % 2 == 0)
    fams.append(1)
    hyps.append(NUMS % 2 == 1)
    fams.append(1)
    for k in range(3, 13):
        if k == 10:
            continue  # identical to "ends in 0"
        hyps.append(NUMS % k == 0)
        fams.append(2)
    dec = NUMS // 10
    for a in range(11):
        for b in range(a, 11):
            hyps.append((dec >= a) & (dec <= b))
            fams.append(3)
    hyps.append(np.ones(100, dtype=bool))
    fams.append(4)
    return np.array(hyps), np.array(fams)


HYP, FAM = _hypotheses()
N_FAM = 5
FAM_COUNT = np.bincount(FAM, minlength=N_FAM).astype("float64")
LOG_SIZE = np.log(HYP.sum(axis=1).astype("float64"))
# One-hot family membership of each hypothesis, (H, F).
FAM_ONEHOT = np.eye(N_FAM)[FAM]
LOG_WITHIN = -np.log(FAM_COUNT[FAM])


def prepare_observed(rows):
    set_index, sets = {}, []
    sidx, tidx, ratings = [], [], []
    for r in rows:
        s = r["number_set"]
        if s not in set_index:
            set_index[s] = len(sets)
            sets.append([int(x) for x in s.split(",")])
        sidx.append(set_index[s])
        tidx.append(int(r["target"]) - 1)
        ratings.append(int(float(r["rating"])))
    consist = np.zeros((len(sets), HYP.shape[0]))
    n_ex = np.zeros(len(sets))
    for i, xs in enumerate(sets):
        consist[i] = HYP[:, np.array(xs) - 1].all(axis=1)
        n_ex[i] = len(xs)
    return {
        "set_idx": np.array(sidx, dtype="int64"),
        "target_idx": np.array(tidx, dtype="int64"),
        "set_consistent": consist,
        "set_n_examples": n_ex,
        "rating": np.array(ratings, dtype="int64"),
    }


with pm.Model() as model:
    set_idx = pm.Data("set_idx", np.zeros(1, dtype="int64"))
    target_idx = pm.Data("target_idx", np.zeros(1, dtype="int64"))
    consist = pm.Data("set_consistent", np.ones((1, HYP.shape[0]), dtype="float64"))
    n_ex = pm.Data("set_n_examples", np.ones(1, dtype="float64"))

    # Prior plausibility of each concept family, and size-principle strength.
    fam_w = pm.Dirichlet("family_weight", a=np.ones(N_FAM))
    size_strength = pm.LogNormal("size_strength", mu=0.0, sigma=0.5)

    log_prior = pt.dot(pt.log(fam_w), FAM_ONEHOT.T) + LOG_WITHIN  # (H,)
    log_lik = -size_strength * n_ex[:, None] * LOG_SIZE[None, :]  # (S, H)
    log_post = log_prior[None, :] + log_lik + (consist - 1.0) * 1e4
    # Posterior predictive for every number 1-100, once per distinct set.
    w = pt.special.softmax(log_post, axis=1)  # (S, H)
    p_in = pt.clip(pt.dot(w, HYP.astype("float64")), 6e-6, 1 - 6e-6)  # (S, 100)
    post_logit = pt.log(p_in) - pt.log1p(-p_in)
    post_logit = post_logit[set_idx, target_idx]

    bias = pm.Normal("bias", mu=0.0, sigma=1.5)
    slope = pm.HalfNormal("slope", sigma=1.0)
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(bias + slope * post_logit), 1e-6, 1 - 1e-6)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
