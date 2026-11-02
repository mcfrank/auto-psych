"""Pedagogical-sampling concept learning (rational pedagogy).

People do not assume the hidden program produced its examples at random from
its concept; they assume the examples were chosen to be helpful, as a teacher
would choose them. A teacher holding a concept picks each example in
proportion to how strongly that example, on its own, would lead a learner with
the same prior to that concept, and people judge each candidate concept by how
likely such a teacher would be to have picked the examples shown. The concept
space and prior are those of the best model (45 rules in families with their
own prior mass, bounded ranges with an Erlang prior on width, one-sided
thresholds as their own family), and "yes" is the posterior probability that
the inferred concept contains the target. So the most diagnostic examples
count most: a concept whose examples could equally well point to many other
concepts is discounted, even if it is small.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
ALPHA = 0.9


def rule_concepts():
    rules = {
        "odd": [n for n in NUMBERS if n % 2 == 1],
        "primes": [n for n in NUMBERS if n > 1 and all(n % a for a in range(2, int(n**0.5) + 1))],
        "squares": [n * n for n in range(1, 11)],
        "cubes": [n**3 for n in range(1, 5)],
        "powers of 2, not 32": [2, 4, 8, 16, 64],
        "powers of 2, or 37": [2, 4, 8, 16, 32, 37, 64],
        "even": list(range(2, 101, 2)),
    }
    rules.update({f"multiples of {m}": list(range(m, 101, m)) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]})
    rules.update({f"powers of {m}": [m**k for k in range(1, 8) if m**k <= 100] for m in range(2, 11)})
    rules.update({f"ends in {d}": [n for n in NUMBERS if n % 10 == d] for d in range(10)})
    rules.update({f"contains digit {d}": [n for n in NUMBERS if str(d) in str(n)] for d in range(10)})
    return rules


def family(name):
    if name in ("odd", "even"):
        return 0
    if name.startswith("multiples"):
        return 2
    if name.startswith("ends"):
        return 3
    if name.startswith("contains"):
        return 4
    if name.startswith("powers of 2,"):
        return 5  # gerrymandered exception rules
    return 1  # primes, squares, cubes, powers


_RULES = rule_concepts()
RULES = list(_RULES.values())
N_RULES = len(RULES)
RULE_FAMILY = np.array([family(n) for n in _RULES])
FAMILY_SIZE = np.bincount(RULE_FAMILY).astype("float64")
LO, HI = np.array([(m, n) for m in NUMBERS for n in range(m, 101)]).T
THRESHOLD = (LO == 1) | (HI == 100)
WIDTH = HI - LO + 1
WIDTHS = np.arange(1, 99)
N_PER_WIDTH = np.array([np.sum(~THRESHOLD & (WIDTH == w)) for w in WIDTHS], dtype="float64")
RANGE_WIDTH_INDEX = np.clip(WIDTH[~THRESHOLD] - 1, 0, len(WIDTHS) - 1)
# All concepts: rules, then bounded ranges, then thresholds.
MEMBERSHIP = np.concatenate(
    [
        np.array([np.isin(NUMBERS, c) for c in RULES]),
        (NUMBERS[None, :] >= LO[~THRESHOLD, None]) & (NUMBERS[None, :] <= HI[~THRESHOLD, None]),
        (NUMBERS[None, :] >= LO[THRESHOLD, None]) & (NUMBERS[None, :] <= HI[THRESHOLD, None]),
    ]
).astype("float64")
N_CONCEPTS, N_RANGES, N_THRESHOLD = len(MEMBERSHIP), int((~THRESHOLD).sum()), float(THRESHOLD.sum())
SIZE = MEMBERSHIP.sum(axis=1)

with pm.Model() as model:
    example_count = pm.Data("example_count", np.zeros((1, 100)))  # (sets, numbers)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    family_weight = pm.Dirichlet("family_weight", a=np.ones(8))
    log_w = pt.log(family_weight)
    interval_scale = pm.LogNormal("interval_scale", mu=np.log(20.0), sigma=1.0)

    erlang = np.log(WIDTHS) - WIDTHS / interval_scale
    log_per_range = erlang - pt.logsumexp(erlang + np.log(N_PER_WIDTH))
    log_prior = pt.concatenate(
        [
            (log_w[:6] - np.log(FAMILY_SIZE))[RULE_FAMILY],
            log_w[6] + log_per_range[RANGE_WIDTH_INDEX],
            pt.repeat(log_w[7] - np.log(N_THRESHOLD), int(N_THRESHOLD)),
        ]
    )
    # How strongly one example x on its own leads a learner with this prior to a
    # concept h containing it: P(h | x) = P(h) / |h| / Z(x). The teacher picks the
    # examples of h in proportion to it, i.e. in proportion to 1 / Z(x) over x in h.
    # Examples are 90% chosen by the teacher, 10% at random from 1..100.
    inv_z = 1.0 / pt.dot(pt.exp(log_prior) / SIZE, MEMBERSHIP)  # (numbers,)
    teach = MEMBERSHIP * inv_z[None, :] / pt.dot(MEMBERSHIP, inv_z)[:, None]
    log_example = pt.log(ALPHA * teach + (1 - ALPHA) / 100)  # (concepts, numbers)
    log_lik = pt.dot(example_count, log_example.T)  # (sets, concepts)
    log_joint = log_prior[None, :] + log_lik
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = pt.dot(posterior, MEMBERSHIP).reshape((-1,))  # (sets x targets)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    count = np.zeros((len(unique), 100))
    for i, s in enumerate(unique):
        np.add.at(count[i], np.array(s) - 1, 1.0)
    return {
        "example_count": count,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
