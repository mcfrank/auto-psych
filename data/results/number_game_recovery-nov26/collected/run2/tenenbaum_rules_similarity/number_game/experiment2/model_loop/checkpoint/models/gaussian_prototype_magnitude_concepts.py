"""People do the same Bayesian concept learning as extended_multiples_concepts for
rules (45 independently weighted rules plus multiples of 13..50, an inferred share
of random examples, rule examples drawn without replacement), but a "range of
magnitudes" is not a sharp-edged interval: it is a fuzzy magnitude prototype, a
centre c with a spread w, from which examples are drawn with Gaussian probability
over 1..100 (the size principle acting through w). A target is judged by its
similarity to the prototype, exp(-(t - c)^2 / (2 w^2)), so for examples that fit
no rule "yes" is strongest near the middle of the examples and fades smoothly with
numerical distance, with no cut-off at the outermost example.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_EX = 4


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
N_BASE = len(_RULES)
EXT_M = list(range(13, 51))
N_EXT = len(EXT_M)
_RULES.update({f"multiples of {m}": list(range(m, 101, m)) for m in EXT_M})
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)

# Magnitude prototypes: centres on a half-integer grid, spreads log-spaced.
CENTRES = np.arange(1.0, 101.0)
WIDTHS = np.exp(np.linspace(np.log(0.8), np.log(40.0), 12))
N_W = len(WIDTHS)
PC = np.repeat(CENTRES, N_W)
PW = np.tile(WIDTHS, len(CENTRES))
PW_IDX = np.tile(np.arange(N_W), len(CENTRES))
N_PROTO = len(PC)
_D = NUMBERS[None, :] - PC[:, None]
SIMILARITY = np.exp(-(_D**2) / (2 * PW[:, None] ** 2))  # (protos, numbers), peak 1 at centre
GEN = SIMILARITY / SIMILARITY.sum(axis=1, keepdims=True)  # example-generation probability
LOG_W = np.log(WIDTHS)

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_BASE - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

with pm.Model() as model:
    rule_n_in = pm.Data("rule_n_in", np.zeros((1, N_RULES)))  # (sets, rules)
    n_examples = pm.Data("n_examples", np.ones(1))  # (sets,)
    ex_gen = pm.Data("ex_gen", np.ones((1, N_PROTO)))  # (examples, protos): generation prob
    ex_set = pm.Data("ex_set", np.zeros(1, dtype="int64"))  # (examples,) its set
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_target = pm.Data("cell_target", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
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
    ext_share = pm.Beta("ext_share", 1.0, 4.0)
    all_weights = pt.concatenate([(1 - ext_share) * rule_weights, pt.ones(N_EXT) * ext_share / N_EXT])

    # Prior over prototype spread: log-normal on the grid, typical spread inferred.
    log_spread = pm.Normal("log_spread", mu=np.log(6.0), sigma=1.0)
    log_wprior = -((LOG_W - log_spread) ** 2) / 2.0
    log_wprior = log_wprior - pt.logsumexp(log_wprior) - np.log(len(CENTRES))

    log_out = pt.log(noise / 100.0)
    rule_log_lik = (n_examples[:, None] - rule_n_in) * log_out
    for j in range(MAX_EX):
        rule_term = pt.log((1 - noise) / np.maximum(RULE_SIZE - j, 1.0) + noise / 100.0)
        rule_log_lik = rule_log_lik + pt.gt(rule_n_in, j) * rule_term[None, :]
    ex_log_lik = pt.log((1 - noise) * ex_gen + noise / 100.0)
    proto_log_lik = pt.zeros((n_examples.shape[0], N_PROTO))
    proto_log_lik = pt.inc_subtensor(proto_log_lik[ex_set], ex_log_lik)

    log_rules = pt.log(lambda_0) + pt.log(all_weights)[None, :] + rule_log_lik
    log_protos = (pt.log1p(-lambda_0) + log_wprior[PW_IDX])[None, :] + proto_log_lik
    log_joint = pt.concatenate([log_rules, log_protos], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    proto_sim = pt.dot(posterior[:, N_RULES:], SIMILARITY)  # (sets, numbers)
    cell_post = posterior[cell_set]
    in_concept = (cell_post[:, :N_RULES] * cell_rule_member).sum(axis=1) + proto_sim[cell_set, cell_target]

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set`, `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    rule_n_in = np.zeros((S, N_RULES))
    n_ex = np.zeros(S)
    gen, ex_set = [], []
    for i, s in enumerate(unique):
        ex = np.array(s) - 1
        n_ex[i] = len(s)
        rule_n_in[i] = RULE_MEMBERSHIP[:, ex].sum(axis=1)
        for x in ex:
            gen.append(GEN[:, x])
            ex_set.append(i)
    keys = [(index[s], int(r["target"]) - 1) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {k: i for i, k in enumerate(cells)}
    cs = np.array([k[0] for k in cells], dtype="int64")
    ct = np.array([k[1] for k in cells], dtype="int64")
    return {
        "rule_n_in": rule_n_in,
        "n_examples": n_ex,
        "ex_gen": np.array(gen),
        "ex_set": np.array(ex_set, dtype="int64"),
        "cell_set": cs,
        "cell_target": ct,
        "cell_rule_member": RULE_MEMBERSHIP[:, ct].T.copy(),
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
