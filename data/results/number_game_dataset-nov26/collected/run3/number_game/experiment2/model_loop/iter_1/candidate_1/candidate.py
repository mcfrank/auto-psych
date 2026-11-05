"""Inattentive-guess lapse on weak-sampling hypothesis elimination.

People infer the hidden concept by simple hypothesis elimination: they keep
every rule (one of 45 mathematical or digit rules, each with its own prior
plausibility) or range of numbers consistent with the examples (an example
that does not fit counts only as a possible stray, at inferred rate eps), with
no size principle. The one claim of this model is at the decision rule: on an
inferred share `lapse` of trials people do not consult that belief at all and
answer by an inattentive guess, saying "yes" with an inferred guessing bias
`guess_yes`. So even examples are endorsed short of certainty and firmly
rejected targets keep a floor of yeses; every judgment is compressed toward
the guessing bias by the same proportion.
"""


import numpy as np
import pymc as pm
import pytensor.tensor as pt

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
    return rules


RULES = list(rule_concepts().values())
N_RULES = len(RULES)
RULE_MEM = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
INT_MEM = np.array([(NUMBERS >= m) & (NUMBERS <= n) for m in NUMBERS for n in range(m, 101)],
                   dtype="float64")
N_INT = INT_MEM.shape[0]
ALL_MEM = np.concatenate([RULE_MEM, INT_MEM])  # (H, 100)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    # Number of examples each concept fails to contain (weak sampling: only this matters).
    misses = np.stack([(1.0 - ALL_MEM[:, np.array(s) - 1]).sum(axis=1) for s in unique])
    cell_keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(cell_keys))
    cell_index = {c: i for i, c in enumerate(cells)}
    cell_flat = np.array([c[0] * 100 + c[1] - 1 for c in cells], dtype="int64")
    return {
        "concept_misses": misses,
        "cell_flat": cell_flat,
        "row_cell": np.array([cell_index[c] for c in cell_keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    concept_misses = pm.Data("concept_misses", np.zeros((1, N_RULES + N_INT)))
    cell_flat = pm.Data("cell_flat", np.zeros(1, dtype="int64"))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)                      # prior share of rules vs ranges
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    eps = pm.Beta("eps", 1.5, 4.0)                            # belief an example is a stray

    log_prior = pt.concatenate([
        pt.log(lambda_0) + pt.log(rule_weights),
        pt.full((N_INT,), pt.log1p(-lambda_0) - np.log(N_INT)),
    ])
    logpost = log_prior[None, :] + concept_misses * pt.log(eps)
    posterior = pt.exp(logpost - pt.logsumexp(logpost, axis=1, keepdims=True))
    in_concept = pt.dot(posterior, pt.as_tensor(ALL_MEM)).reshape((-1,))[cell_flat]

    lapse = pm.Beta("lapse", 2.0, 4.0)                        # share of inattentive guesses
    guess_yes = pm.Beta("guess_yes", 2.0, 2.0)                # bias toward "yes" when guessing
    p_raw = (1.0 - lapse) * in_concept[row_cell] + lapse * guess_yes
    p_yes = pm.Deterministic("p_yes", pt.clip(p_raw, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)
