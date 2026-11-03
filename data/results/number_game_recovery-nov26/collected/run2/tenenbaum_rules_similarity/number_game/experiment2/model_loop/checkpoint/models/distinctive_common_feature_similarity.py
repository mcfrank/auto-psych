"""People judge the target by its similarity to the example set as a whole,
using a feature-based (Tversky-style common-features) metric: each arithmetic
or digit property the target shares with the examples adds to similarity in
proportion to how consistently the examples share it (fraction of examples
with the property, raised to a power gamma) and how distinctive it is (rarer
properties among 1-100 count more, exponent beta); closeness on the number
line to the span of the examples adds a magnitude-band match that decays
exponentially with distance. P(yes) is a logistic function of the log summed
similarity. No inference over a single generating concept is made.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)


def rule_concepts():
    rules = {
        "odd": [n for n in NUMBERS if n % 2 == 1],
        "even": list(range(2, 101, 2)),
        "primes": [n for n in NUMBERS if n > 1 and all(n % a for a in range(2, int(n**0.5) + 1))],
        "squares": [n * n for n in range(1, 11)],
        "cubes": [n**3 for n in range(1, 5)],
    }
    rules.update({f"multiples of {m}": list(range(m, 101, m)) for m in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]})
    rules.update({f"powers of {m}": [m**k for k in range(1, 8) if m**k <= 100] for m in range(2, 11)})
    rules.update({f"ends in {d}": [n for n in NUMBERS if n % 10 == d] for d in range(10)})
    rules.update({f"contains digit {d}": [n for n in NUMBERS if str(d) in str(n)] for d in range(10)})
    return list(rules.values())


RULES = rule_concepts()
N_RULES = len(RULES)
MEMBER = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")  # (rules, 100)
LOG_RARITY = np.log(100.0 / MEMBER.sum(axis=1))  # distinctiveness of each feature


with pm.Model() as model:
    # Per (set, target) cell: fraction of examples sharing each feature the
    # target has (0 if the target lacks it), and distance to the examples' span.
    shared_frac = pm.Data("shared_frac", np.zeros((1, N_RULES)))
    span_dist = pm.Data("span_dist", np.zeros(1))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    gamma = pm.LogNormal("gamma", mu=np.log(3.0), sigma=0.7)  # demand for consistency
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.7)  # weight of distinctiveness
    tau = pm.LogNormal("tau", mu=np.log(0.08), sigma=0.7)  # magnitude-band decay (1-100 scale / 100)
    log_w_mag = pm.Normal("log_w_mag", 0.0, 1.5)  # weight of the magnitude-band match
    intercept = pm.Normal("intercept", 0.0, 2.0)
    slope = pm.LogNormal("slope", mu=0.0, sigma=0.7)

    feat = pt.switch(shared_frac > 0, pt.exp(gamma * pt.log(pt.maximum(shared_frac, 1e-9))), 0.0)
    feat_sim = (feat * pt.exp(beta * LOG_RARITY)[None, :]).sum(axis=1)
    mag_sim = pt.exp(log_w_mag - span_dist / tau)
    sim = feat_sim + mag_sim
    p_cell = pm.math.sigmoid(intercept + slope * pt.log(sim + 1e-3))
    p_yes = pm.Deterministic("p_yes", pt.clip(p_cell[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    keys = [(tuple(int(x) for x in str(r["number_set"]).split(",")), int(r["target"])) for r in rows]
    cells = list(dict.fromkeys(keys))
    index = {k: i for i, k in enumerate(cells)}
    shared = np.zeros((len(cells), N_RULES))
    dist = np.zeros(len(cells))
    for i, (s, t) in enumerate(cells):
        ex = np.array(s) - 1
        frac = MEMBER[:, ex].mean(axis=1)
        shared[i] = frac * MEMBER[:, t - 1]
        dist[i] = max(min(s) - t, t - max(s), 0) / 100.0
    return {
        "shared_frac": shared,
        "span_dist": dist,
        "row_cell": np.array([index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
