"""Exemplar account: people store each example separately and judge the target
by its average similarity to the stored exemplars. The similarity of target t to
exemplar x is one-shot generalisation: how probable it is that a familiar
concept (45 rules with a family-organised prior, exception rules their own kind,
or a magnitude range with a beta prior over its share of 1..100) that produced x
also contains t, with specific (small) concepts favoured by the size principle.
The examples are never combined into one jointly-consistent concept; the pooled
similarity, raised to an inferred sensitivity exponent, is the probability of
answering yes.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

ALPHA = 0.9
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
FAMILY = np.array([rule_family(n) for n in RULE_NAMES])
N_FAMILIES = FAMILY.max() + 1
FAMILY_SIZE = np.bincount(FAMILY)
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
EXAMPLE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]  # (rules, numbers)
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]
INTERVAL_SIZE_IDX = (INTERVAL_MEMBERSHIP.sum(axis=1) - 1).astype("int64")
SIZES = np.arange(1, 101, dtype="float64")
LOG_N_OF_SIZE = np.log(101.0 - SIZES)
SHARE = (SIZES - 0.5) / 100.0

# One-shot tables, per single exemplar x: rule log-likelihood (x, rules), per-size
# interval log evidence (x, sizes), and per size the share of that evidence from
# intervals containing each target (x, sizes, targets). All parameter-free.
ONE_RULE_LL = EXAMPLE_LOG_LIK[:N_RULES].T.copy()
ONE_SIZE_EV = np.zeros((100, 100))
ONE_SIZE_MEMBER = np.zeros((100, 100, 100))
for _L in range(100):
    _sel = INTERVAL_SIZE_IDX == _L
    _ll = EXAMPLE_LOG_LIK[N_RULES:][_sel]  # (intervals of size L, x)
    _ev = logsumexp(_ll, axis=0)  # (x,)
    ONE_SIZE_EV[:, _L] = _ev
    ONE_SIZE_MEMBER[:, _L, :] = np.exp(_ll - _ev[None, :]).T @ INTERVAL_MEMBERSHIP[_sel]
ONE_SIZE_MEMBER = np.clip(ONE_SIZE_MEMBER, 0.0, 1.0)

with pm.Model() as model:
    cell_ex = pm.Data("cell_ex", np.zeros((1, MAX_EX), dtype="int64"))  # exemplar number - 1
    cell_mask = pm.Data("cell_mask", np.ones((1, MAX_EX)))  # 1 for a real exemplar
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))  # target in rule
    cell_size_member = pm.Data("cell_size_member", np.zeros((1, MAX_EX, 100)))  # per exemplar, size
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    mean_share = pm.Beta("mean_share", 2, 3)
    concentration = pm.LogNormal("concentration", mu=np.log(5.0), sigma=1.0)
    family_weights = pm.Dirichlet("family_weights", a=np.full(N_FAMILIES, 2.0))
    sensitivity = pm.LogNormal("sensitivity", mu=0.0, sigma=0.7)

    rule_log_prior = pt.log(family_weights)[FAMILY] - np.log(FAMILY_SIZE[FAMILY])
    log_size_prior = ((mean_share * concentration - 1.0) * np.log(SHARE)
                      + ((1.0 - mean_share) * concentration - 1.0) * np.log1p(-SHARE))
    log_size_prior = log_size_prior - pt.logsumexp(log_size_prior)
    log_per_interval = log_size_prior - LOG_N_OF_SIZE

    # One-shot generalisation from each possible exemplar x (100 x concepts).
    log_rules = pt.log(lambda_0) + rule_log_prior[None, :] + ONE_RULE_LL
    log_sizes = pt.log1p(-lambda_0) + log_per_interval[None, :] + ONE_SIZE_EV
    log_joint = pt.concatenate([log_rules, log_sizes], axis=1)
    post = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))  # (100, concepts)

    ex_post = post[cell_ex]  # (cells, MAX_EX, concepts)
    sim = ((ex_post[:, :, :N_RULES] * cell_rule_member[:, None, :]).sum(axis=2)
           + (ex_post[:, :, N_RULES:] * cell_size_member).sum(axis=2))  # (cells, MAX_EX)
    mean_sim = (sim * cell_mask).sum(axis=1) / cell_mask.sum(axis=1)
    mean_sim = pt.clip(mean_sim, 1e-6, 1 - 1e-6)
    p_cell = pt.exp(sensitivity * pt.log(mean_sim))

    p_yes = pm.Deterministic("p_yes", pt.clip(p_cell[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    keys = []
    for r in rows:
        ex = tuple(int(x) for x in str(r["number_set"]).split(","))
        keys.append((ex, int(r["target"])))
    cells = list(dict.fromkeys(keys))
    cell_index = {c: i for i, c in enumerate(cells)}
    n = len(cells)
    ex_idx = np.zeros((n, MAX_EX), dtype="int64")
    mask = np.zeros((n, MAX_EX))
    size_member = np.zeros((n, MAX_EX, 100))
    tgt = np.zeros(n, dtype="int64")
    for i, (ex, t) in enumerate(cells):
        tgt[i] = t - 1
        for k, x in enumerate(ex[:MAX_EX]):
            ex_idx[i, k] = x - 1
            mask[i, k] = 1.0
            size_member[i, k] = ONE_SIZE_MEMBER[x - 1, :, t - 1]
    return {
        "cell_ex": ex_idx,
        "cell_mask": mask,
        "cell_rule_member": RULE_MEMBERSHIP[:, tgt].T.copy(),
        "cell_size_member": size_member,
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
