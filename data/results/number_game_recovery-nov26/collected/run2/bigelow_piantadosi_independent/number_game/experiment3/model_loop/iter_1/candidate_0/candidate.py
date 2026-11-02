"""Pedagogical (helpful-teacher) sampling in Bayesian concept learning.

People treat the examples as chosen on purpose by a helpful teacher rather than
drawn at random from the hidden concept: they weigh each concept (45 arithmetic
rules with kind plausibilities and sparse within-kind familiarities, and every
interval of 1-100) by how likely a teacher who wanted them to grasp that
concept would have been to pick exactly these examples. The teacher picks an
example x for concept h in proportion to how strongly x alone would lead a
literal (randomly sampling) learner with the same prior to h, raised to a
helpfulness exponent alpha:

    p_teach(x | h) ∝ [lik(x | h) / sum_h' prior(h') lik(x | h')] ** alpha.

Examples that single a concept out among the alternatives are therefore
evidence for it, beyond the size principle. Evidence from several examples is
pooled with a capacity limit n ** (-gamma), and "yes" is the posterior
probability that the concept contains the target.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

STRAY = 0.9
NUMBERS = np.arange(1, 101)


def rule_concepts():
    """(family, members) for each rule; families index FAMILIES."""
    primes = [n for n in NUMBERS if n > 1 and all(n % a for a in range(2, int(n**0.5) + 1))]
    rules = [
        (0, [n for n in NUMBERS if n % 2 == 1]),
        (1, list(range(2, 101, 2))),
        (2, primes),
        (3, [n * n for n in range(1, 11)]),
        (3, [n**3 for n in range(1, 5)]),
        (4, [2, 4, 8, 16, 64]),
        (4, [2, 4, 8, 16, 32, 37, 64]),
    ]
    rules += [(8 if m == 5 else 5, list(range(m, 101, m))) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]]
    rules += [(4, [m**k for k in range(1, 8) if m**k <= 100]) for m in range(2, 11)]
    rules += [(6, [n for n in NUMBERS if n % 10 == d]) for d in range(10)]
    rules += [(7, [n for n in NUMBERS if str(d) in str(n)]) for d in range(10)]
    return rules


FAMILIES = ["odd", "even", "primes", "squares_cubes", "powers", "multiples", "final_digit", "contains_digit", "multiples_of_5"]
_RULES = rule_concepts()
RULE_FAMILY = np.array([f for f, _ in _RULES], dtype="int64")
N_RULES = len(_RULES)
N_FAM = len(FAMILIES)
FAMILY_COUNT = np.bincount(RULE_FAMILY, minlength=N_FAM)
FAMILY_LOG_COUNT = np.log(FAMILY_COUNT).astype("float64")
KIND_ONEHOT = (RULE_FAMILY[:, None] == np.arange(N_FAM)[None, :]).astype("float64")
FAMILIARITY_SPREAD = 2.5
MULTI_RULES = np.flatnonzero(FAMILY_COUNT[RULE_FAMILY] > 1)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for _, c in _RULES], dtype="float64")
INTERVALS = np.array([(m, n) for m in NUMBERS for n in range(m, 101)])
N_INTERVALS = len(INTERVALS)
INTERVAL_MEMBERSHIP = ((NUMBERS[None, :] >= INTERVALS[:, :1]) & (NUMBERS[None, :] <= INTERVALS[:, 1:])).astype("float64")
MEMBERSHIP = np.concatenate([RULE_MEMBERSHIP, INTERVAL_MEMBERSHIP])  # (concepts, 100)
# Random-sampling likelihood of one example under each concept (10% strays).
EX_LIK = STRAY * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - STRAY) / 100
EX_LL = np.log(EX_LIK)
N_CONCEPTS = MEMBERSHIP.shape[0]
MEMBERSHIP32 = MEMBERSHIP.astype("float32")
RULE_EX_LIK = EX_LIK[:N_RULES]
# Intervals share one prior, so their summed likelihood of each number is a constant;
# an interval's example likelihood takes one value inside it and one outside.
INTERVAL_LIK_SUM = EX_LIK[N_RULES:].sum(axis=0)  # (100,)
INTERVAL_LL_IN = np.log(STRAY / (INTERVALS[:, 1] - INTERVALS[:, 0] + 1) + (1 - STRAY) / 100)
LL_OUT = np.log((1 - STRAY) / 100)
INTERVAL_LO = INTERVALS[:, 0] - 1  # prefix-sum index of the first member
INTERVAL_HI = INTERVALS[:, 1]  # prefix-sum index past the last member

with pm.Model() as model:
    set_lit_ll = pm.Data("set_lit_ll", np.zeros((1, N_CONCEPTS)))  # (sets, concepts) random-sampling log-lik of the set
    set_n = pm.Data("set_n", np.ones(1))  # number of examples per set
    set_log_n = pm.Data("set_log_n", np.zeros(1))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_target = pm.Data("cell_target", np.zeros(1, dtype="int64"))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    family_logit = pm.Normal("family_logit", mu=0.0, sigma=2.5, shape=N_FAM)
    familiarity_raw = pm.StudentT("familiarity_z", nu=5.0, mu=0.0, sigma=0.4, shape=len(MULTI_RULES))
    familiarity_z = pt.set_subtensor(pt.zeros(N_RULES)[MULTI_RULES], familiarity_raw)
    kind_mean_z = (pt.dot(KIND_ONEHOT.T, familiarity_z) / FAMILY_COUNT)[RULE_FAMILY]
    log_rule_raw = family_logit[RULE_FAMILY] - FAMILY_LOG_COUNT[RULE_FAMILY] + FAMILIARITY_SPREAD * (familiarity_z - kind_mean_z)
    log_norm = pt.logsumexp(pt.concatenate([log_rule_raw, pt.zeros(1)]))
    log_prior = pt.concatenate([log_rule_raw - log_norm, pt.zeros(N_INTERVALS) - log_norm - np.log(N_INTERVALS)])

    # Teacher helpfulness: how strongly examples are chosen to single out the concept.
    alpha = pm.LogNormal("alpha", mu=0.0, sigma=0.7)
    gamma = pm.Normal("gamma", mu=0.0, sigma=0.5)

    # Literal learner's marginal probability of each single example.
    marg = pt.dot(pt.exp(log_prior[:N_RULES]), RULE_EX_LIK) + pt.exp(log_prior[N_RULES]) * INTERVAL_LIK_SUM
    log_marg = pt.log(marg)  # (100,)
    # log p_teach(x | h) = alpha * EX_LL[h, x] - alpha * log_marg[x] - teach_norm[h]; the
    # middle term is the same for every concept, so it cancels in the posterior.
    rule_norm = pt.logsumexp(alpha * (EX_LL[:N_RULES] - log_marg[None, :]), axis=1)
    weight = pt.exp(-alpha * (log_marg - pt.max(log_marg)))  # (100,), rescaled for stability
    prefix = pt.concatenate([pt.zeros(1), pt.cumsum(weight)])
    sum_in = prefix[INTERVAL_HI] - prefix[INTERVAL_LO]
    sum_out = pt.maximum(prefix[100] - sum_in, 0.0)
    interval_norm = pt.log(pt.exp(alpha * (INTERVAL_LL_IN - LL_OUT)) * sum_in + sum_out) + alpha * (LL_OUT - pt.max(log_marg))
    teach_norm = pt.concatenate([rule_norm, interval_norm])  # (concepts,)

    capacity = pt.exp(-gamma * set_log_n)[:, None]
    evidence = alpha * set_lit_ll - set_n[:, None] * teach_norm[None, :]  # (sets, concepts)
    log_joint = log_prior[None, :] + capacity * evidence
    # The (sets x concepts) posterior is held in single precision for speed.
    log_joint32 = pt.cast(log_joint - pt.max(log_joint, axis=1, keepdims=True), "float32")
    weight_post = pt.exp(log_joint32)  # unnormalized posterior
    p_target32 = pt.dot(weight_post, MEMBERSHIP32) / weight_post.sum(axis=1, keepdims=True)
    p_target = pt.cast(p_target32, "float64")  # (sets, 100)
    in_concept = p_target[cell_set, cell_target]

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept, 1e-6, 1 - 1e-6)[row_cell])
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    set_index = {s: i for i, s in enumerate(unique)}
    lit_ll = np.stack([EX_LL[:, np.array(s) - 1].sum(axis=1) for s in unique])
    cells = list(dict.fromkeys((s, int(r["target"])) for s, r in zip(sets, rows)))
    cell_index = {c: k for k, c in enumerate(cells)}
    return {
        "set_lit_ll": lit_ll,
        "set_n": np.array([len(s) for s in unique], dtype="float64"),
        "set_log_n": np.log(np.array([len(s) for s in unique], dtype="float64")),
        "cell_set": np.array([set_index[s] for s, _ in cells], dtype="int64"),
        "cell_target": np.array([t - 1 for _, t in cells], dtype="int64"),
        "row_cell": np.array([cell_index[(s, int(r["target"]))] for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(float(r.get("rating", 0))) for r in rows], dtype="int64"),
    }
