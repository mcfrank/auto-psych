"""Times-table rule sampling.

People do not compare the target with each example by similarity; on each
trial they come up with one guess about the program and answer by whether the
target obeys it. The guess is a familiar property every example has (odd/even,
multiples of 3-12, a shared final digit, squares, cubes, powers of two) or,
failing that, "numbers in the region the examples cover" (their span with a
soft margin). Each kind of property comes to mind with its own ease (smaller
times tables easier), and properties grow more likely than the region guess
with every further example that fits them. A small lapse rate gives guesses.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMS = np.arange(1, 101)
RULES = []  # (type, kfeat, membership over 1..100)
RULES.append((0, 0.0, NUMS % 2 == 1))
RULES.append((0, 0.0, NUMS % 2 == 0))
for k in range(3, 13):
    RULES.append((1, np.log(k / 6.0), NUMS % k == 0))
for d in range(10):
    RULES.append((2, 0.0, NUMS % 10 == d))
RULES.append((3, 0.0, np.isin(NUMS, [i * i for i in range(1, 11)])))
RULES.append((3, 0.0, np.isin(NUMS, [i ** 3 for i in range(1, 5)])))
RULES.append((3, 0.0, np.isin(NUMS, [2 ** i for i in range(0, 7)])))
R = len(RULES)
RTYPE = np.array([r[0] for r in RULES], dtype="int64")
KFEAT = np.array([r[1] for r in RULES], dtype="float64")
MEMB = np.stack([r[2] for r in RULES]).astype("float64")  # (R, 100)


def prepare_observed(rows):
    n = len(rows)
    cons = np.zeros((n, R))
    mem = np.zeros((n, R))
    hull = np.zeros(n)
    nex = np.zeros(n)
    for i, r in enumerate(rows):
        ex = [int(x) for x in str(r["number_set"]).split(",")]
        t = int(r["target"])
        cons[i] = MEMB[:, np.array(ex) - 1].min(axis=1)
        mem[i] = MEMB[:, t - 1]
        hull[i] = max(0, min(ex) - t, t - max(ex)) / 10.0
        nex[i] = len(ex) - 1
    return {
        "consistent": cons,
        "member": mem,
        "hull_dist": hull,
        "extra_examples": nex,
        "rating": np.array([int(r.get("rating", 0)) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    consistent = pm.Data("consistent", np.zeros((1, R), dtype="float64"))
    member = pm.Data("member", np.zeros((1, R), dtype="float64"))
    hull_dist = pm.Data("hull_dist", np.zeros(1, dtype="float64"))
    extra = pm.Data("extra_examples", np.zeros(1, dtype="float64"))

    # Ease of thinking of each kind of property, relative to the region guess.
    s_type = pm.Normal("s_type", mu=0.0, sigma=2.0, shape=4)
    table_slope = pm.Normal("table_slope", mu=0.0, sigma=2.0)
    accumulation = pm.Normal("accumulation", mu=0.0, sigma=2.0)
    log_margin = pm.Normal("log_margin", mu=0.0, sigma=1.0)
    lapse = pm.Beta("lapse", alpha=2.0, beta=8.0)
    guess_yes = pm.Beta("guess_yes", alpha=2.0, beta=2.0)

    rule_logit = (s_type[RTYPE] + table_slope * KFEAT)[None, :] + accumulation * extra[:, None]
    rule_w = pt.exp(rule_logit) * consistent  # only properties all examples have
    region_w = pt.ones_like(hull_dist)
    total = pt.sum(rule_w, axis=1) + region_w
    region_member = pt.exp(-hull_dist / pt.exp(log_margin))
    p_rule = (pt.sum(rule_w * member, axis=1) + region_w * region_member) / total

    p = lapse * guess_yes + (1.0 - lapse) * p_rule
    p_yes = pm.Deterministic("p_yes", pt.clip(p, 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
