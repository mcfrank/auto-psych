"""Win-stay, lose-shift concept tracking (primacy through a sticky first guess).

People hold a single working guess about the concept (one of 45 rules or an
interval of 1..100). After the first example the guess is drawn from the
Bayesian posterior (size principle, 10% noisy examples, a prior weight per
rule, an Erlang prior on interval size). As each further example arrives,
if it lies in the current guess the learner keeps it with probability
`stay` (no size-principle re-weighting); otherwise (contradiction, or not
sticking) the guess is redrawn from the posterior given all examples so far.
"Yes" is the probability that the final guess contains the target. With
stay = 0 this is the full posterior learner; stay > 0 makes early examples
weigh more (a primacy / anchoring effect).
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
NUMBERS = np.arange(1, 101)
MAX_N = 4


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
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
N_HYP = MEMBERSHIP.shape[0]
EXAMPLE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
INTERVAL_SIZE = MEMBERSHIP[N_RULES:].sum(axis=1)

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

with pm.Model() as model:
    prefix_log_lik = pm.Data("prefix_log_lik", np.zeros((1, MAX_N, N_HYP)))  # log lik of examples 1..k
    contains = pm.Data("contains", np.ones((1, MAX_N, N_HYP)))  # example k lies in hypothesis
    step_mask = pm.Data("step_mask", np.zeros((1, MAX_N)))  # example k exists
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set index * 100 + target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    set_weights = pm.Dirichlet("set_weights", a=SET_CONCENTRATION)
    zero_split = pm.Beta("zero_split", 2, 2)
    shared = set_weights[END_0]
    rule_weights = pt.concatenate([
        set_weights[:END_0], (zero_split * shared)[None], set_weights[END_0 + 1:CONTAINS_0],
        ((1 - zero_split) * shared)[None], set_weights[CONTAINS_0:],
    ])
    log_sigma = pm.Normal("log_sigma", mu=np.log(10.0), sigma=1.0)
    size = pt.as_tensor(INTERVAL_SIZE)
    log_erlang = pt.log(size) - size / pt.exp(log_sigma)
    log_prior = pt.concatenate([
        pt.log(lambda_0) + pt.log(rule_weights),
        pt.log1p(-lambda_0) + log_erlang - pt.logsumexp(log_erlang),
    ])

    # Probability of keeping a guess that the new example fits.
    stay = pm.Beta("stay", 2, 2)

    log_joint = log_prior[None, None, :] + prefix_log_lik
    post = pt.exp(log_joint - pt.logsumexp(log_joint, axis=2, keepdims=True))  # (sets, steps, hyp)

    q = post[:, 0, :]
    for k in range(1, MAX_N):
        kept = stay * q * contains[:, k, :]
        updated = kept + (1 - kept.sum(axis=1, keepdims=True)) * post[:, k, :]
        m = step_mask[:, k][:, None]
        q = m * updated + (1 - m) * q

    in_concept = q @ pt.as_tensor(MEMBERSHIP)  # (sets, 100)
    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    prefix = np.zeros((len(unique), MAX_N, N_HYP))
    contains = np.ones((len(unique), MAX_N, N_HYP))
    mask = np.zeros((len(unique), MAX_N))
    for i, s in enumerate(unique):
        running = np.zeros(N_HYP)
        for k in range(MAX_N):
            if k < len(s):
                running = running + EXAMPLE_LOG_LIK[:, s[k] - 1]
                contains[i, k] = MEMBERSHIP[:, s[k] - 1]
                mask[i, k] = 1.0
            prefix[i, k] = running
    return {
        "prefix_log_lik": prefix,
        "contains": contains,
        "step_mask": mask,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
