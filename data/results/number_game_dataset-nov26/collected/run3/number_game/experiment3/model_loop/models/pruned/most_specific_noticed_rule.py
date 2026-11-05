"""Most-specific noticed rule, with a range fallback (satisficing rule search).

People do not weigh many candidate concepts at once; they search for a single
rule and stop at the first one they notice. Each familiar property that every
example shares (parity, multiples of k, a special number kind such as squares,
powers or primes, a final digit, a contained digit) is noticed with a
probability that depends on its kind; among the noticed shared properties
people adopt the most specific (smallest) one and say "yes" to targets that
have it. Only when no shared property is noticed do they fall back on the
examples' span, endorsing targets inside it and fewer the farther a target lies
outside it. A response floor/ceiling reflects unsystematic answering.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)


def _isprime(n):
    return n > 1 and all(n % a for a in range(2, int(n ** 0.5) + 1))


def _rules():
    rules, fam = [], []
    rules.append([n for n in NUMBERS if n % 2]); fam.append(0)
    rules.append([n for n in NUMBERS if n % 2 == 0]); fam.append(0)
    for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]:
        rules.append(list(range(m, 101, m))); fam.append(1)
    rules.append([n for n in NUMBERS if _isprime(n)]); fam.append(2)
    rules.append([n * n for n in range(1, 11)]); fam.append(2)
    rules.append([n ** 3 for n in range(1, 5)]); fam.append(2)
    for m in range(2, 11):
        rules.append([m ** k for k in range(1, 8) if m ** k <= 100]); fam.append(2)
    for d in range(10):
        rules.append([n for n in NUMBERS if n % 10 == d]); fam.append(3)
    for d in range(10):
        rules.append([n for n in NUMBERS if str(d) in str(n)]); fam.append(4)
    mem = np.array([np.isin(NUMBERS, r) for r in rules], dtype="float64")
    fam = np.array(fam, dtype="int64")
    order = np.argsort(mem.sum(axis=1), kind="stable")  # most specific first
    return mem[order], fam[order]


RULE_MEM, RULE_FAM = _rules()
N_RULES, N_FAM = RULE_MEM.shape[0], 5


def prepare_observed(rows):
    sets = sorted({r["number_set"] for r in rows})
    index = {s: i for i, s in enumerate(sets)}
    fits = np.zeros((len(sets), N_RULES))
    lo = np.zeros(len(sets)); hi = np.zeros(len(sets))
    for s, i in index.items():
        ex = np.array([int(x) for x in s.split(",")])
        fits[i] = RULE_MEM[:, ex - 1].prod(axis=1)
        lo[i], hi[i] = ex.min(), ex.max()
    set_idx = np.array([index[r["number_set"]] for r in rows], dtype="int64")
    t = np.array([int(r["target"]) for r in rows], dtype="int64")
    dist = np.maximum(lo[set_idx] - t, 0) + np.maximum(t - hi[set_idx], 0)
    return {
        "set_fits": fits,
        "trial_set": set_idx,
        "target_membership": RULE_MEM[:, t - 1].T.copy(),
        "outside_distance": dist.astype("float64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    set_fits = pm.Data("set_fits", np.zeros((1, N_RULES)))
    trial_set = pm.Data("trial_set", np.zeros(1, dtype="int64"))
    target_membership = pm.Data("target_membership", np.zeros((1, N_RULES)))
    outside_distance = pm.Data("outside_distance", np.zeros(1))

    # Probability of noticing a shared property, by kind:
    # parity, multiples, special numbers, final digit, contained digit.
    notice = pm.Beta("notice", 1.0, 1.0, shape=N_FAM)
    tau = pm.LogNormal("tau", mu=np.log(5.0), sigma=1.0)  # range-fallback falloff
    floor = pm.Beta("floor", 1.0, 6.0)   # "yes" regardless of belief
    miss = pm.Beta("miss", 1.0, 3.0)     # "no" regardless of belief

    qf = notice[RULE_FAM][None, :] * set_fits           # sets x rules
    log_skip = pt.log1p(-qf * (1.0 - 1e-9))
    cum = pt.cumsum(log_skip, axis=1) - log_skip         # none smaller noticed
    p_adopt = qf * pt.exp(cum)
    p_none = pt.exp(pt.sum(log_skip, axis=1))

    rule_part = pt.sum(p_adopt[trial_set] * target_membership, axis=1)
    range_part = p_none[trial_set] * pt.exp(-outside_distance / tau)
    belief = rule_part + range_part

    p_yes = pm.Deterministic(
        "p_yes", pt.clip(floor + (1.0 - floor) * (1.0 - miss) * belief, 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
