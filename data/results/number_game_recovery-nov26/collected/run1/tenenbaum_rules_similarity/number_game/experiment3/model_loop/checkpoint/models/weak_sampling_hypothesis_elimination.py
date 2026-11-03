"""Weak-sampling hypothesis elimination.

People entertain the same space of candidate programs as the best concept
learner -- number rules (parity, primes, squares, cubes, powers, multiples,
final and contained digits, digit sums, two exception rules), each kind with
its own prior plausibility, and numeric ranges with a prior over width -- but
they treat each example only as a test the program passed, not as a sample the
program generated. Programs that fail to contain an example are (largely)
eliminated, while every program consistent with the examples keeps its prior
share, however large it is: there is no "suspicious coincidence" (size
principle). "Yes" is the prior-weighted share of the surviving programs that
contain the target, so a consistent broad rule such as "even" competes with
"powers of 2" on prior plausibility alone, and ranges are narrowed only by the
width prior, not by the examples' spread.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

# Probability an example's positive label reflects the program (else a coin flip).
ALPHA = 0.9
NUMBERS = np.arange(1, 101)


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
    rules.update({f"digit sum {t}": [n for n in NUMBERS if sum(map(int, str(n))) == t] for t in range(1, 19)})
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
    if name.startswith("digit sum"):
        return 6
    if name.startswith("powers of 2,"):
        return 5  # gerrymandered exception rules
    return 1  # primes, squares, cubes, powers


_RULES = rule_concepts()
RULES = list(_RULES.values())
N_RULES = len(RULES)
RULE_FAMILY = np.array([family(n) for n in _RULES])
FAMILY_SIZE = np.bincount(RULE_FAMILY).astype("float64")
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
LO, HI = np.array([(m, n) for m in NUMBERS for n in range(m, 101)]).T
WIDTH = HI - LO + 1
INTERVAL_MEMBERSHIP = ((NUMBERS[None, :] >= LO[:, None]) & (NUMBERS[None, :] <= HI[:, None])).astype("float64")
MEMBERSHIP = np.concatenate([RULE_MEMBERSHIP, INTERVAL_MEMBERSHIP])  # (hypotheses, 100)
WIDTHS = np.arange(1, 101, dtype="float64")
N_PER_WIDTH = 101.0 - WIDTHS
# Weak sampling: P(example labelled positive | program) does not depend on the
# program's size -- ALPHA if it contains the example, plus a coin-flip lapse.
LOG_CONSISTENT = np.log(ALPHA + (1 - ALPHA) / 2)
LOG_INCONSISTENT = np.log((1 - ALPHA) / 2)

with pm.Model() as model:
    # Per set, the weak-sampling likelihood of the examples under each rule:
    # (sets, rules).
    rule_lik = pm.Data("rule_lik", np.zeros((1, N_RULES)))
    # Ranges of equal width share a prior, so per set and width the summed
    # likelihood of all ranges of that width, (sets, widths), and the same sum
    # restricted to ranges containing each target, (sets, widths, targets).
    width_lik = pm.Data("width_lik", np.zeros((1, 100)))
    width_target_lik = pm.Data("width_target_lik", np.zeros((1, 100, 100)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Prior mass of the seven rule families (incl. exception rules and digit
    # sums) and of ranges; equal shares within a family.
    family_weight = pm.Dirichlet("family_weight", a=np.ones(8))
    # Characteristic width of the Erlang prior over range width.
    interval_scale = pm.LogNormal("interval_scale", mu=np.log(20.0), sigma=1.0)

    erlang = np.log(WIDTHS) - WIDTHS / interval_scale
    per_range = pt.exp(erlang - pt.logsumexp(erlang + np.log(N_PER_WIDTH)))  # prior of one range of width w
    range_prior = family_weight[7] * per_range  # (widths,)
    rule_prior = family_weight[:7][RULE_FAMILY] / FAMILY_SIZE[RULE_FAMILY]  # (rules,)

    rule_joint = rule_lik * rule_prior[None, :]  # (sets, rules)
    evidence = pt.sum(rule_joint, axis=1) + pt.dot(width_lik, range_prior)  # (sets,)
    yes_mass = pt.dot(rule_joint, pt.as_tensor(RULE_MEMBERSHIP)) + pt.tensordot(
        width_target_lik, range_prior, axes=[[1], [0]]
    )  # (sets, targets)
    in_concept = yes_mass / evidence[:, None]

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    n_rules = len(RULE_MEMBERSHIP)
    rule_lik = np.zeros((len(unique), n_rules))
    width_lik = np.zeros((len(unique), 100))
    width_target_lik = np.zeros((len(unique), 100, 100))
    for i, s in enumerate(unique):
        log_lik = np.zeros(len(MEMBERSHIP))
        for x in s:
            log_lik += np.where(MEMBERSHIP[:, x - 1] > 0, LOG_CONSISTENT, LOG_INCONSISTENT)
        lik = np.exp(log_lik)
        rule_lik[i] = lik[:n_rules]
        range_lik = lik[n_rules:]
        np.add.at(width_lik[i], WIDTH - 1, range_lik)
        np.add.at(width_target_lik[i], WIDTH - 1, range_lik[:, None] * INTERVAL_MEMBERSHIP)
    return {
        "rule_lik": rule_lik,
        "width_lik": width_lik,
        "width_target_lik": width_target_lik,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
