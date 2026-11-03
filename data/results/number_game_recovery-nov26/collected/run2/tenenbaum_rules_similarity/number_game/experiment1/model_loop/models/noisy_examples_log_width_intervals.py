"""Refinement of log_scale_interval_width_prior: Bayesian concept learning over
45 independently weighted rules and all intervals of 1..100 (size principle),
with an Erlang prior on each interval's width measured on a logarithmic mental
number line, L = log((hi + 0.5) / (lo - 0.5)), weight proportional to
L * exp(-L / tau). The one change, grafted from inferred_example_noise_concepts,
is that the share `noise` of examples believed drawn uniformly from 1..100 is
inferred rather than fixed at 10%. "Yes" is the posterior probability that the
concept contains the target.
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
INTERVAL_BOUNDS = [(m, n) for m in NUMBERS for n in range(m, 101)]
INTERVALS = [list(range(m, n + 1)) for m, n in INTERVAL_BOUNDS]
# Width of each interval on the log number line.
LOG_WIDTH = np.array([np.log((n + 0.5) / (m - 0.5)) for m, n in INTERVAL_BOUNDS])
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
CONCEPT_SIZE = MEMBERSHIP.sum(axis=1)

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

with pm.Model() as model:
    n_in = pm.Data("n_in", np.zeros((1, len(CONCEPT_SIZE))))  # (sets, concepts): examples inside
    n_examples = pm.Data("n_examples", np.ones(1))  # (sets,)
    # Only the (set, target) cells that were asked about are evaluated.
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # (cells,)
    cell_member = pm.Data("cell_member", np.zeros((1, len(CONCEPT_SIZE))))  # (cells, concepts)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # (trials,) index into cells
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    set_weights = pm.Dirichlet("set_weights", a=SET_CONCENTRATION)
    zero_split = pm.Beta("zero_split", 2, 2)
    shared = set_weights[END_0]
    rule_weights = pm.Deterministic("rule_weights", pt.concatenate([
        set_weights[:END_0], (zero_split * shared)[None], set_weights[END_0 + 1:CONTAINS_0],
        ((1 - zero_split) * shared)[None], set_weights[CONTAINS_0:],
    ]))
    noise = pm.Beta("noise", 1.5, 6.0)  # share of examples believed random
    # Preferred log-width of a range: Erlang(2, tau) over log(hi / lo).
    tau = pm.LogNormal("tau", mu=np.log(0.5), sigma=1.0)
    log_erlang = pt.log(LOG_WIDTH) - LOG_WIDTH / tau
    log_per_interval = log_erlang - pt.logsumexp(log_erlang)
    log_prior = pt.concatenate([pt.log(lambda_0) + pt.log(rule_weights), pt.log1p(-lambda_0) + log_per_interval])

    log_in = pt.log((1 - noise) / CONCEPT_SIZE + noise / 100.0)  # (concepts,)
    log_out = pt.log(noise / 100.0)
    log_lik = n_in * log_in[None, :] + (n_examples[:, None] - n_in) * log_out
    log_joint = log_prior[None, :] + log_lik
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = (posterior[cell_set] * cell_member).sum(axis=1)  # (cells,)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    keys = [(index[s], int(r["target"]) - 1) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {k: i for i, k in enumerate(cells)}
    return {
        "n_in": np.stack([MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "n_examples": np.array([float(len(s)) for s in unique]),
        "cell_set": np.array([k[0] for k in cells], dtype="int64"),
        "cell_member": MEMBERSHIP[:, [k[1] for k in cells]].T.copy(),
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
