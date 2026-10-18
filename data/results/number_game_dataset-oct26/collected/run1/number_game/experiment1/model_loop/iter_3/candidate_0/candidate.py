"""People infer that the program counts in fixed steps: the concepts they consider
are arithmetic progressions through 1-100 (all numbers with the same remainder
modulo a step size d, d = 1..99), preferring small steps a priori but, by the size
principle, favouring the largest step all the examples fit. They say "yes" with the
posterior probability that the target lies on the inferred progression."""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

STEPS = np.arange(1, 100)
NUMS = np.arange(1, 101)


def prepare_observed(rows):
    sets = sorted({r["number_set"] for r in rows})
    sidx = {s: i for i, s in enumerate(sets)}
    S, K = len(sets), len(STEPS)
    logsize = np.zeros((S, K))
    consistent = np.zeros((S, K))
    n_ex = np.zeros(S)
    residues = np.zeros((S, K), dtype=int)
    for i, s in enumerate(sets):
        ex = [int(x) for x in s.split(",")]
        n_ex[i] = len(ex)
        for k, d in enumerate(STEPS):
            r = ex[0] % d
            residues[i, k] = r
            consistent[i, k] = float(all(e % d == r for e in ex))
            logsize[i, k] = np.log(np.sum(NUMS % d == r))
    member = np.zeros((len(rows), K))
    set_idx = np.zeros(len(rows), dtype="int64")
    for t, r in enumerate(rows):
        i = sidx[r["number_set"]]
        set_idx[t] = i
        member[t] = (int(r["target"]) % STEPS == residues[i]).astype(float)
    # set-level (n_ex*logsize, consistent mask) broadcast per trial for pm.Data
    return {
        "member": member,
        "set_idx": set_idx,
        "set_neglik": n_ex[:, None] * logsize,
        "set_mask": np.where(consistent > 0, 0.0, -1e4),
        "rating": np.array([int(float(r["rating"])) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    member = pm.Data("member", np.zeros((1, len(STEPS))))
    set_idx = pm.Data("set_idx", np.zeros(1, dtype="int64"))
    set_neglik = pm.Data("set_neglik", np.zeros((1, len(STEPS))))
    set_mask = pm.Data("set_mask", np.zeros((1, len(STEPS))))

    # prior preference for small steps: weight proportional to d^-alpha
    alpha = pm.HalfNormal("alpha", sigma=3.0)
    # strength of the size principle (1 = each example a random draw)
    size_w = pm.HalfNormal("size_w", sigma=1.0)
    guess = pm.Beta("guess", 1.0, 5.0)
    lapse = pm.Beta("lapse", 1.0, 5.0)

    logpost = -alpha * np.log(STEPS)[None, :] - size_w * set_neglik + set_mask
    post = pt.special.softmax(logpost, axis=1)
    p_in = pt.sum(post[set_idx] * member, axis=1)
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(guess + (1.0 - guess - lapse) * p_in, 1e-6, 1 - 1e-6)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
