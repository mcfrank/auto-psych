"""Magnitude prototypes on the number line in place of inferred ranges.

People do not infer a bounded numeric range with sharp edges when they
generalize by magnitude. They abstract a prototype: a central number on the
number line, with a Gaussian similarity gradient of some width around it. They
take the program to produce numbers scattered around its prototype, with
nearer numbers more likely (a discretised Gaussian, with 10% stray numbers).
Prototype locations (1-100) and gradient widths (a log-spaced grid from 0.5 to
60, equal prior) are weighed by how well they explain the examples. A target is
endorsed by its expected similarity to the prototype. Clustered examples favour
a tight prototype at their centre, spread-out examples a broad one, and the
examples themselves are endorsed less than fully when they sit off-centre.
Arithmetic rules compete with the prototypes as in the best model: 42 rules,
a rule prior rising as a power of size with uneven idiosyncratic familiarity,
perceived rule sizes inflated by a constant, tempered evidence, and a decision
gain with a shifted criterion.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

EPSILON = 0.1
NUMBERS = np.arange(1, 101)


def rule_sets():
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
    return rules


RULES = rule_sets()
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
RULE_LOG_SIZE_C = np.log(RULE_SIZE) - np.log(RULE_SIZE).mean()
SIZE_DIR = RULE_LOG_SIZE_C / np.linalg.norm(RULE_LOG_SIZE_C)
FAMILIARITY_SPREAD = 3.0
LOG_NOISE = np.log(EPSILON / 100)

# Prototype similarity widths entertained (equal prior over a log-spaced grid).
WIDTHS = np.geomspace(0.5, 60.0, 16)
N_W = len(WIDTHS)
LOG_W_PRIOR = -np.log(N_W) - np.log(100)  # width x centre, uniform
DIST = np.abs(NUMBERS[:, None] - NUMBERS[None, :]).astype("float64")  # (100, 100)
SIM = np.exp(-0.5 * (DIST[None, :, :] / WIDTHS[:, None, None]) ** 2)  # (widths, centre, number)
KERNEL = SIM / SIM.sum(axis=2, keepdims=True)  # generation kernel around a prototype


def prototype_log_lik(examples):
    """(width*centre,) log-probability of the examples around each prototype."""
    ex = np.array(examples) - 1
    ll = np.log((1 - EPSILON) * KERNEL[:, :, ex] + EPSILON / 100).sum(axis=2)  # (widths, centres)
    return ll.reshape(-1)


with pm.Model() as model:
    rule_inside = pm.Data("rule_inside", np.zeros((1, N_RULES)))  # (sets, rules)
    set_size = pm.Data("set_size", np.ones(1))  # (sets,)
    prototype_ll = pm.Data("prototype_ll", np.zeros((1, N_W * 100)))  # (sets, widths*centres)
    prototype_sim = pm.Data("prototype_sim", np.zeros((1, N_W * 100, 1)))  # (sets, widths*centres, asked targets)
    set_rules = pm.Data("set_rules", np.zeros((1, N_RULES, 1)))  # (sets, rules, asked targets)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 4, 4)
    familiarity_z = pm.ZeroSumNormal("familiarity_z", sigma=1.0, shape=N_RULES)
    familiarity = FAMILIARITY_SPREAD * (familiarity_z - pt.dot(familiarity_z, SIZE_DIR) * SIZE_DIR)
    size_exponent = pm.Normal("size_exponent", mu=0.0, sigma=0.5)
    rule_evidence_weight = pm.LogNormal("rule_evidence_weight", mu=0.0, sigma=0.5)
    prototype_evidence_weight = pm.LogNormal("prototype_evidence_weight", mu=0.0, sigma=0.5)
    size_inflation = pm.Deterministic(
        "size_inflation", pt.softplus(pm.Normal("size_inflation_raw", mu=2.0, sigma=1.5))
    )
    log_gain = pm.Normal("log_gain", mu=0.0, sigma=0.5)
    bias = pm.Normal("bias", mu=0.0, sigma=1.0)

    n = set_size[:, None]
    rule_in = pt.log((1 - EPSILON) / (pt.as_tensor(RULE_SIZE) + size_inflation) + EPSILON / 100)[None, :]
    rule_ev = rule_inside * rule_in + (n - rule_inside) * LOG_NOISE
    log_rule_prior = familiarity + size_exponent * RULE_LOG_SIZE_C
    log_rule_prior = log_rule_prior - pt.logsumexp(log_rule_prior)

    log_rules = pt.log(p_mix) + log_rule_prior[None, :] + rule_evidence_weight * rule_ev
    log_proto = pt.log1p(-p_mix) + LOG_W_PRIOR + prototype_evidence_weight * prototype_ll
    log_joint = pt.concatenate([log_rules, log_proto], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    cell_p = pt.batched_dot(posterior[:, :N_RULES], set_rules) + pt.batched_dot(posterior[:, N_RULES:], prototype_sim)
    concept_p = pt.clip(cell_p.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6)

    belief_logit = pt.log(concept_p) - pt.log1p(-concept_p)
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(pt.exp(log_gain) * belief_logit + bias), 1e-6, 1 - 1e-6)
    )
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    n_sets = len(unique)
    set_idx = np.array([index[s] for s in sets], dtype="int64")
    tgt = np.array([int(r["target"]) - 1 for r in rows], dtype="int64")
    asked = [set() for _ in range(n_sets)]
    for si, t in zip(set_idx.tolist(), tgt.tolist()):
        asked[si].add(t)
    asked = [sorted(a) for a in asked]
    t_max = max(len(a) for a in asked)
    pad = [a + [a[0]] * (t_max - len(a)) for a in asked]
    pos = [{t: k for k, t in enumerate(a)} for a in asked]
    # Similarity of each asked target to every prototype (width, centre).
    sim = np.stack([SIM[:, :, pad[i]].reshape(N_W * 100, -1) for i in range(n_sets)])
    return {
        "rule_inside": np.stack([RULE_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "set_size": np.array([float(len(s)) for s in unique]),
        "prototype_ll": np.stack([prototype_log_lik(s) for s in unique]),
        "prototype_sim": sim,
        "set_rules": np.stack([RULE_MEMBERSHIP[:, pad[i]] for i in range(n_sets)]),
        "row_cell": np.array([si * t_max + pos[si][t] for si, t in zip(set_idx.tolist(), tgt.tolist())], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
