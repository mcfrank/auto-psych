"""Memoryless win-stay, lose-shift rule testing over the examples in order.

People hold one candidate rule at a time. The first example suggests a rule
(final digit d, multiples of g, odd, or a range of sizes in blocks of 4; smaller rules
favoured by the size principle). Each later example that fits the rule keeps
it; an example that contradicts it makes people discard it and draw a fresh
rule suggested by that example alone, without re-checking earlier examples.
They answer "yes" if the target obeys the rule held at the end (plus a small
floor/ceiling of guessing). Example order therefore changes generalisation.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAXK = 4
NUMS = np.arange(1, 101)


def _build_hypotheses():
    members, family = [], []
    for d in range(10):                     # ends in d
        members.append(NUMS % 10 == d)
        family.append(0)
    for g in range(2, 13):                  # multiples of g (g=2: even)
        members.append(NUMS % g == 0)
        family.append(1)
    members.append(NUMS % 2 == 1)           # odd
    family.append(1)
    for a in range(1, 101, 4):              # intervals [a, b] in blocks of 4
        for b in range(a + 3, 101, 4):
            members.append((NUMS >= a) & (NUMS <= b))
            family.append(2)
    return np.array(members, dtype="float64"), np.array(family)


MEMBER, FAMILY = _build_hypotheses()        # (H, 100), (H,)
H = MEMBER.shape[0]
SIZE = MEMBER.sum(axis=1)
LOG_SIZE = np.log(SIZE)
N_FAM = np.array([np.sum(FAMILY == f) for f in range(3)], dtype="float64")
IS_INT = (FAMILY == 2).astype("float64")


def prepare_observed(rows):
    sets = []
    index = {}
    set_idx = np.zeros(len(rows), dtype="int64")
    tgt = np.zeros(len(rows), dtype="int64")
    rating = np.zeros(len(rows), dtype="int64")
    for r, row in enumerate(rows):
        s = str(row["number_set"])
        if s not in index:
            index[s] = len(sets)
            sets.append([int(x) for x in s.split(",")][:MAXK])
        set_idx[r] = index[s]
        tgt[r] = int(row["target"]) - 1
        rating[r] = int(float(row["rating"]))
    cons = np.zeros((len(sets), MAXK, H))
    step = np.zeros((len(sets), MAXK))
    for i, ex in enumerate(sets):
        for j, x in enumerate(ex):
            cons[i, j] = MEMBER[:, x - 1]
            step[i, j] = 1.0
        for j in range(len(ex), MAXK):
            cons[i, j] = 1.0                # absent step: everything "fits"
    return {"consistent": cons, "step_mask": step, "set_idx": set_idx,
            "target_idx": tgt, "rating": rating}


with pm.Model() as model:
    consistent = pm.Data("consistent", np.ones((1, MAXK, H)))
    step_mask = pm.Data("step_mask", np.ones((1, MAXK)))
    set_idx = pm.Data("set_idx", np.zeros(1, dtype="int64"))
    target_idx = pm.Data("target_idx", np.zeros(1, dtype="int64"))

    # Prior plausibility of rule families (digit, multiples/parity, interval).
    fam_logit = pm.Normal("fam_logit", mu=0.0, sigma=1.5, shape=2)
    # Typical interval size (Erlang prior over interval width).
    int_scale = pm.LogNormal("int_scale", mu=np.log(15.0), sigma=0.75)
    guess_yes = pm.Beta("guess_yes", alpha=2.0, beta=10.0)   # floor of "yes"
    guess_no = pm.Beta("guess_no", alpha=2.0, beta=10.0)     # ceiling of "no"

    fam_lp = pt.log(pm.math.softmax(pt.concatenate([pt.zeros(1), fam_logit])))
    erlang = LOG_SIZE - 2.0 * pt.log(int_scale) - SIZE / int_scale
    erlang_norm = erlang - pt.logsumexp(erlang * IS_INT - 1e9 * (1 - IS_INT))
    log_prior = pt.switch(IS_INT > 0, fam_lp[2] + erlang_norm,
                          fam_lp[FAMILY] - np.log(N_FAM[FAMILY]))
    # Rule suggested by an example: prior x size-principle likelihood.
    w = pt.exp(log_prior - LOG_SIZE - pt.max(log_prior - LOG_SIZE))  # (H,)

    def suggest(c):
        q = w[None, :] * c
        return q / pt.sum(q, axis=1, keepdims=True)

    belief = suggest(consistent[:, 0, :])
    for j in range(1, MAXK):
        c = consistent[:, j, :]
        lose = pt.sum(belief * (1.0 - c), axis=1, keepdims=True)
        new = belief * c + lose * suggest(c)
        m = step_mask[:, j][:, None]
        belief = m * new + (1.0 - m) * belief

    p_member = pt.dot(belief, MEMBER)                 # (sets, 100)
    p_rule = p_member[set_idx, target_idx]
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(guess_yes + (1.0 - guess_yes - guess_no) * p_rule,
                         1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
