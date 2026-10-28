"""Pedagogical (helpful-teacher) sampling of examples.

People assume the examples were chosen by a helpful teacher rather than drawn
at random from the concept: for each example, the teacher is thought to pick a
member of the concept with probability rising, by an inferred power (pedagogy
strength), with how strongly that single example would point a naive learner
(with no preference among rules or among ranges) to the concept. The learner
inverts this teacher model over 42 weighted rules and all intervals of 1..100
(10% noisy examples, evidence tempered separately for rules and ranges) and
answers through a decision gain and shifted criterion. Pedagogy strength zero
recovers ordinary strong sampling (the size principle).

Speed: the naive learner is parameter-free, so how diagnostic each number is
(log Z) is precomputed; only one per-concept normaliser depends on the
pedagogy strength.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

EPSILON = 0.1
NUMBERS = np.arange(1, 101)
MAX_SET = 4


def concept_sets():
    """42 rules, then 5,050 intervals."""
    rules = [
        [2, 4, 8, 16, 32, 64], [3, 9, 27, 81], [4, 16, 64], [5, 25], [6, 36],
        [7, 49], [8, 64], [9, 81], [10, 100],
    ]
    rules += [[i for i in NUMBERS if i % n == 0] for n in range(3, 13)]
    rules += [[i for i in NUMBERS if i % 10 == d] for d in range(1, 10)]
    rules += [[i for i in NUMBERS if str(d) in str(i)] for d in range(1, 10)]
    rules.append([1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97])
    rules.append([i for i in NUMBERS if i % 2 == 0])
    rules.append([i for i in NUMBERS if i % 2 == 1])
    rules.append([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])
    rules.append([1, 8, 27, 64])
    intervals = [list(range(m, n + 1)) for m in range(1, 101) for n in range(m, 101)]
    return rules, intervals


RULES, INTERVALS = concept_sets()
N_RULES = len(RULES)
N_INT = len(INTERVALS)
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")  # (H, 100)
SIZE = MEMBERSHIP.sum(axis=1, keepdims=True)
IS_RULE = np.r_[np.ones(N_RULES), np.zeros(N_INT)]

# Naive learner: half belief on rules (equal), half on ranges (equal); strong
# sampling with noise. After one example x its posterior on a concept h that
# contains x is prior(h) * lik(x | h) / Z(x), so the only part that varies across
# the members of h is -log Z(x): a number is diagnostic when few, large concepts
# contain it. The teacher therefore picks member x of h with probability
# w(x) / sum_{x' in h} w(x'), where w(x) = exp(-pedagogy * log Z(x)).
_naive_prior = np.where(IS_RULE == 1, 0.5 / N_RULES, 0.5 / N_INT)
_naive = _naive_prior[:, None] * (MEMBERSHIP * (1 - EPSILON) / SIZE + EPSILON / 100)
LOG_Z = np.log(_naive.sum(axis=0))
LOG_Z = LOG_Z - LOG_Z.mean()  # centred: (100,)


with pm.Model() as model:
    pair_number = pm.Data("pair_number", np.zeros(1, dtype="int64"))  # example (0-based) of each pair
    pair_concept = pm.Data("pair_concept", np.zeros(1, dtype="int64"))  # concept containing it
    pair_cell = pm.Data("pair_cell", np.zeros(1, dtype="int64"))  # set index * H + concept
    set_size = pm.Data("set_size", np.zeros(1))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set index * 100 + target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 4, 4)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    rule_evidence_weight = pm.LogNormal("rule_evidence_weight", mu=0.0, sigma=0.5)
    range_evidence_weight = pm.LogNormal("range_evidence_weight", mu=0.0, sigma=0.5)
    log_gain = pm.Normal("log_gain", mu=0.0, sigma=0.5)
    bias = pm.Normal("bias", mu=0.0, sigma=1.0)
    # The mechanism: how strongly the teacher is believed to favour diagnostic examples.
    pedagogy = pm.Normal("pedagogy", mu=0.0, sigma=1.0)

    w = pt.exp(-pedagogy * pt.as_tensor(LOG_Z))  # (100,)
    inv_norm = 1.0 / pt.dot(pt.as_tensor(MEMBERSHIP), w)  # (H,)
    # Only (example, concept) pairs where the concept contains the example carry
    # the teacher's choice; noise: log((1-e) t + e/100) = log(e/100) + log1p(t (1-e) 100 / e).
    pair_lik = pt.log1p((w * ((1 - EPSILON) * 100 / EPSILON))[pair_number] * inv_norm[pair_concept])
    set_lik = pt.zeros(set_size.shape[0] * (N_RULES + N_INT))
    set_lik = pt.inc_subtensor(set_lik[pair_cell], pair_lik).reshape((-1, N_RULES + N_INT))
    set_lik = set_lik + set_size[:, None] * np.log(EPSILON / 100)

    log_prior = pt.concatenate(
        [pt.log(p_mix) + pt.log(rule_weights), pt.ones(N_INT) * (pt.log1p(-p_mix) - np.log(N_INT))]
    )
    evidence_weight = rule_evidence_weight * IS_RULE + range_evidence_weight * (1 - IS_RULE)
    log_joint = log_prior[None, :] + evidence_weight[None, :] * set_lik
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))  # (sets, H)
    concept_p = pt.dot(posterior, pt.as_tensor(MEMBERSHIP)).reshape((-1,))[row_cell]
    concept_p = pt.clip(concept_p, 1e-6, 1 - 1e-6)

    belief_logit = pt.log(concept_p) - pt.log1p(-concept_p)
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(pt.exp(log_gain) * belief_logit + bias), 1e-6, 1 - 1e-6)
    )
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    ex = np.zeros((len(unique), MAX_SET), dtype="int64")
    mask = np.zeros((len(unique), MAX_SET))
    for i, s in enumerate(unique):
        ex[i, : len(s)] = np.array(s) - 1
        mask[i, : len(s)] = 1.0
    pair_number, pair_concept, pair_cell = [], [], []
    for i, s in enumerate(unique):
        for x in s:
            h = np.flatnonzero(MEMBERSHIP[:, x - 1])
            pair_number.append(np.full(h.size, x - 1, dtype="int64"))
            pair_concept.append(h.astype("int64"))
            pair_cell.append((i * (N_RULES + N_INT) + h).astype("int64"))
    set_idx = np.array([index[s] for s in sets], dtype="int64")
    tgt = np.array([int(r["target"]) - 1 for r in rows], dtype="int64")
    return {
        "pair_number": np.concatenate(pair_number),
        "pair_concept": np.concatenate(pair_concept),
        "pair_cell": np.concatenate(pair_cell),
        "set_size": mask.sum(axis=1),
        "row_cell": set_idx * 100 + tgt,
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
