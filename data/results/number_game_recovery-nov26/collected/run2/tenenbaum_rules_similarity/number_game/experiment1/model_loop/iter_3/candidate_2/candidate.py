"""Pedagogical sampling: people assume the examples were chosen by a helpful
teacher, not drawn at random from the concept. The teacher prefers examples that
are diagnostic -- numbers that few other plausible concepts also contain -- so
an example x of concept h is chosen with probability proportional to
lik0(x|h) * Z(x)^-alpha, where lik0 is the size-principle likelihood (an
inferred share `noise` of examples random over 1..100) and Z(x) the marginal
probability of x under the learner's prior. alpha = 0 is the random-sampling
learner; alpha > 0 penalises concepts full of distinctive numbers that were not
shown. The hypothesis space and prior are the incumbent's: 45 independently
weighted rules (share lambda_0) and every interval of 1..100 with an Erlang prior
on width. "Yes" is the posterior probability that the concept contains the target.
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


_RULES = rule_concepts()
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")

# Every interval [m, n] of 1..100 as an explicit hypothesis (its position matters
# for how diagnostic its members are, so intervals cannot be grouped by size).
IV = [(m, n) for m in range(1, 101) for n in range(m, 101)]
IV_MEMBERSHIP = np.array([(NUMBERS >= m) & (NUMBERS <= n) for m, n in IV], dtype="float64")
IV_SIZE = IV_MEMBERSHIP.sum(axis=1)
SIZES_F = np.arange(1, 101, dtype="float64")
N_PER_SIZE = 101.0 - SIZES_F
IV_SIZE_IDX = (IV_SIZE - 1).astype("int64")

MEMBERSHIP = np.concatenate([RULE_MEMBERSHIP, IV_MEMBERSHIP])  # (hypotheses, 100)
H_SIZE = MEMBERSHIP.sum(axis=1)
N_H = len(H_SIZE)

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

with pm.Model() as model:
    n_in = pm.Data("n_in", np.zeros((1, N_H)))  # (sets, hypotheses): examples inside
    n_examples = pm.Data("n_examples", np.ones(1))  # (sets,)
    row_set = pm.Data("row_set", np.zeros(1, dtype="int64"))  # (trials,)
    row_target = pm.Data("row_target", np.zeros(1, dtype="int64"))  # (trials,) 0-based
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    noise = pm.Beta("noise", 1.5, 6.0)
    set_weights = pm.Dirichlet("set_weights", a=SET_CONCENTRATION)
    zero_split = pm.Beta("zero_split", 2, 2)
    shared = set_weights[END_0]
    rule_weights = pt.concatenate([
        set_weights[:END_0], (zero_split * shared)[None], set_weights[END_0 + 1:CONTAINS_0],
        ((1 - zero_split) * shared)[None], set_weights[CONTAINS_0:],
    ])
    sigma = pm.LogNormal("sigma", mu=np.log(20.0), sigma=1.0)
    log_erlang = pt.log(SIZES_F) - SIZES_F / sigma
    log_per_interval = log_erlang - pt.logsumexp(log_erlang + np.log(N_PER_SIZE))
    log_prior = pt.concatenate([
        pt.log(lambda_0) + pt.log(rule_weights),
        pt.log1p(-lambda_0) + log_per_interval[IV_SIZE_IDX],
    ])  # (hypotheses,)

    # Strength of the assumed teacher's preference for diagnostic examples.
    alpha = pm.HalfNormal("alpha", sigma=1.0)

    # lik0(x|h) = (1 - noise) [x in h] / |h| + noise / 100.
    log_z = pt.log((1 - noise) * pt.dot(pt.exp(log_prior) / H_SIZE, MEMBERSHIP) + noise / 100.0)  # (100,)
    w = pt.exp(-alpha * (log_z - pt.max(log_z)))  # teacher's preference for each number
    log_norm = pt.log((1 - noise) * pt.dot(MEMBERSHIP, w) / H_SIZE + noise / 100.0 * pt.sum(w))  # (H,)
    log_in = pt.log((1 - noise) / H_SIZE + noise / 100.0)
    log_out = pt.log(noise / 100.0)
    # Teacher log-likelihood of the set, up to terms constant across hypotheses.
    log_lik = n_in * (log_in - log_out)[None, :] + n_examples[:, None] * (log_out - log_norm)[None, :]
    log_joint = log_prior[None, :] + log_lik
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))  # (sets, H)
    in_concept = pt.dot(posterior, MEMBERSHIP)  # (sets, 100)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_set, row_target], 1e-9, 1 - 1e-9))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set`, `target` and (for observed data) `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    counts = np.zeros((len(unique), 100))
    for i, s in enumerate(unique):
        for x in s:
            counts[i, x - 1] += 1
    return {
        "n_in": counts @ MEMBERSHIP.T,
        "n_examples": counts.sum(axis=1),
        "row_set": np.array([index[s] for s in sets], dtype="int64"),
        "row_target": np.array([int(r["target"]) - 1 for r in rows], dtype="int64"),
        "rating": np.array([int(r.get("rating", 0)) for r in rows], dtype="int64"),
    }
