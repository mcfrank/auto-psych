"""Similarity-gradient magnitude generalisation: people learn the hidden concept as
the best model does over arithmetic rules (family-organised prior, exception rules
as their own family, size principle with 10% stray examples), but they do not
represent magnitude-based concepts as hard-edged ranges. A magnitude concept is a
neighbourhood around a central number: membership falls off with numerical
distance from that centre along a Gaussian similarity gradient
whose width is the concept's breadth, and examples are produced in proportion to
that similarity. Centre and breadth are inferred from the examples (narrow
neighbourhoods favoured by the size principle, a log-normal prior over breadth
with inferred typical value and spread), and a target's graded similarity to the
inferred neighbourhood is the probability it belongs, so "yes" fades gradually
with distance beyond the examples instead of stopping at a range edge.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

ALPHA = 0.9
NUMBERS = np.arange(1, 101)
WIDTHS = np.exp(np.linspace(np.log(0.5), np.log(60.0), 24))  # breadths of the gradient
LOG_WIDTHS = np.log(WIDTHS)


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


def rule_family(name):
    if name in ("odd", "even"):
        return 0
    if name == "primes":
        return 1
    if name in ("squares", "cubes"):
        return 2
    if name.startswith("powers of 2,"):
        return 7
    if name.startswith("powers"):
        return 3
    if name.startswith("multiples"):
        return 4
    if name.startswith("ends in"):
        return 5
    return 6


_RULES = rule_concepts()
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_LOG_LIK = np.log(ALPHA * RULE_MEMBERSHIP / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
FAMILY = np.array([rule_family(n) for n in RULE_NAMES])
N_FAMILIES = FAMILY.max() + 1
FAMILY_SIZE = np.bincount(FAMILY)

# Neighbourhood concepts: centre c in 1..100, breadth w; similarity of number n
# to the concept is exp(-(n - c)^2 / (2 w^2)). SIM[w, c, n].
_DIST = np.abs(NUMBERS[:, None] - NUMBERS[None, :]).astype("float64")
SIM = np.exp(-0.5 * (_DIST[None, :, :] / WIDTHS[:, None, None]) ** 2)
NEIGH_LOG_LIK = np.log(ALPHA * SIM / SIM.sum(axis=2, keepdims=True) + (1 - ALPHA) / 100)
N_W = len(WIDTHS)

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    width_log_evidence = pm.Data("width_log_evidence", np.zeros((1, N_W)))  # (sets, widths): log mean_c P(D|c,w)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    cell_width_sim = pm.Data("cell_width_sim", np.zeros((1, N_W)))  # E[similarity | D, w]
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    log_width_mu = pm.Normal("log_width_mu", mu=np.log(8.0), sigma=1.0)  # typical breadth
    log_width_sd = pm.LogNormal("log_width_sd", mu=np.log(0.7), sigma=0.5)
    family_weights = pm.Dirichlet("family_weights", a=np.full(N_FAMILIES, 2.0))
    rule_log_prior = pt.log(family_weights)[FAMILY] - np.log(FAMILY_SIZE[FAMILY])

    log_width_prior = -0.5 * ((LOG_WIDTHS - log_width_mu) / log_width_sd) ** 2
    log_width_prior = log_width_prior - pt.logsumexp(log_width_prior)

    log_rules = pt.log(lambda_0) + rule_log_prior[None, :] + rule_log_lik
    log_neigh = pt.log1p(-lambda_0) + log_width_prior[None, :] + width_log_evidence
    log_joint = pt.concatenate([log_rules, log_neigh], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    cell_post = posterior[cell_set]
    in_concept = ((cell_post[:, :N_RULES] * cell_rule_member).sum(axis=1)
                  + (cell_post[:, N_RULES:] * cell_width_sim).sum(axis=1))

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    n_sets = len(unique)
    rule_ll = np.stack([RULE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique])
    width_ev = np.zeros((n_sets, N_W))
    width_sim = np.zeros((n_sets, N_W, 100))  # (sets, widths, targets)
    for i, s in enumerate(unique):
        ll = NEIGH_LOG_LIK[:, :, np.array(s) - 1].sum(axis=2)  # (widths, centres)
        ev = logsumexp(ll, axis=1)
        width_ev[i] = ev - np.log(100.0)  # centre uniform over 1..100
        post_c = np.exp(ll - ev[:, None])
        width_sim[i] = np.einsum("wc,wcn->wn", post_c, SIM)
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: i for i, c in enumerate(cells)}
    cell_s = np.array([c[0] for c in cells], dtype="int64")
    cell_t = np.array([c[1] - 1 for c in cells], dtype="int64")
    return {
        "rule_log_lik": rule_ll,
        "width_log_evidence": width_ev,
        "cell_set": cell_s,
        "cell_rule_member": RULE_MEMBERSHIP[:, cell_t].T.copy(),
        "cell_width_sim": np.clip(width_sim[cell_s, :, cell_t], 0.0, 1.0),
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
