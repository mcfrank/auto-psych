"""Refinement of family_pooled_uncertain_reliability: the same Bayesian concept
learning over 45 rules (log prior weights partially pooled around family levels
with a spread) and every interval of 1-100, with an uncertain
Beta(mean alpha, concentration kappa) belief about how reliable the program is
on each set, integrated out per set. The one change is the sampling
assumption: people are unsure whether the examples were sampled from the
concept (strong sampling, the size principle) or merely produced by the
program and happen to be consistent with it (weak sampling, every number in
1-100 equally likely, so a concept's size gives no evidence), and they average
over the two, with prior probability rho on strong sampling:

p(X | h) = rho * p_strong(X | h) + (1 - rho) * p_weak(X | h),
p_strong = sum_j C(k, j) s^-j 100^-(n-j) B(u + j, v + n - j) / B(u, v),
p_weak   = 100^-n B(u, v + n - k) / B(u, v)  (each non-member must be a stray),

for n examples of which k fall in a concept of size s; u = alpha kappa,
v = (1 - alpha) kappa. Both depend only on (n, k, s), so the table is built once
per gradient step. Unlike a tempered size principle, the mixture lets a set
that a tight concept fits very well switch to strong-sampling reasoning while
a set no tight concept fits keeps broad, size-insensitive generalisation.
"Yes" is the posterior probability that the concept contains the target.

Implementation: the (n, k, s) table is built in probability space and flattened, and per-set / per-(set, target)
interval evidence is a dense dot product with precomputed count matrices (no
gathers on the interval side); family levels are zero-sum, as the softmax over
rules ignores a common shift; the within-family spread of rule log-weights is
stated (1.75, near the incumbent's inferred tau) rather than inferred, which
removes the funnel between the spread and the deviations.
"""

import math

import numpy as np
import pymc as pm
import pytensor.sparse as ps
import pytensor.tensor as pt

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
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_SIZE_IDX = RULE_MEMBERSHIP.sum(axis=1).astype("int64") - 1
INT_LO = np.array([m for m in NUMBERS for n in range(m, 101)])
INT_HI = np.array([n for m in NUMBERS for n in range(m, 101)])
INT_SIZE_IDX = INT_HI - INT_LO  # size - 1

FAMILIES = ["parity", "primes", "squares and cubes", "powers", "multiples", "ends in", "contains digit"]


def family_of(name):
    if name in ("odd", "even"):
        return "parity"
    if name in ("squares", "cubes"):
        return "squares and cubes"
    for f in ("primes", "powers", "multiples", "ends in", "contains digit"):
        if name.startswith(f):
            return f
    raise ValueError(name)


RULE_FAMILY = np.array([FAMILIES.index(family_of(r)) for r in RULE_NAMES])

# Constant parts of the (n, k, s, j) likelihood table; n = 1..4, k, j = 0..4.
_N = np.arange(1, MAX_N + 1, dtype="float64")[:, None, None, None]
_K = np.arange(MAX_N + 1, dtype="float64")[None, :, None, None]
_S = np.arange(1, 101, dtype="float64")[None, None, :, None]
_J = np.arange(MAX_N + 1, dtype="float64")[None, None, None, :]
# Cells with k > n are never read (no set has more members than examples); they
# are kept finite (only j > k is masked) so no gradient through them is NaN.
_VALID = _J <= _K
_LOG_BINOM = np.where(
    _VALID,
    np.vectorize(lambda k, j: math.lgamma(k + 1) - math.lgamma(j + 1) - math.lgamma(max(k - j, 0) + 1))(_K, _J),
    -1e4,
)
_CONST = _LOG_BINOM - _J * np.log(_S) - (_N - _J) * np.log(100.0)
_EXP_CONST = np.exp(_CONST)  # masked (j > k) terms are exactly 0
_N_MINUS_J = np.maximum(_N - _J, 0.0)  # (n, 1, 1, j)
_J_NJ = _J + 0.0 * _N_MINUS_J


def _betaln(a, b):
    return pt.gammaln(a) + pt.gammaln(b) - pt.gammaln(a + b)


_n3 = np.arange(1, MAX_N + 1, dtype="float64")[:, None, None]
_k3 = np.arange(MAX_N + 1, dtype="float64")[None, :, None]
_NK = np.maximum(_n3 - _k3, 0.0)  # (n, k, 1)
_WEAK_CONST = np.exp(-_n3 * np.log(100.0)) + np.zeros((1, 1, 100))


RULE_SPREAD = 1.75  # spread of rule log-weights around their family level
N_CELLS_TABLE = MAX_N * (MAX_N + 1) * 100  # flattened (n, k, size) table


with pm.Model() as model:
    rule_flat_idx = pm.Data("rule_flat_idx", np.zeros((1, N_RULES), dtype="int64"))  # (sets, rules) -> table
    interval_count = pm.Data("interval_count", np.zeros((1, N_CELLS_TABLE)))  # (sets, n*k*size)
    # (cells, n*k*size) counts of intervals containing the target, as CSR parts.
    cc_data = pm.Data("cc_data", np.zeros(1))
    cc_indices = pm.Data("cc_indices", np.zeros(1, dtype="int32"))
    cc_indptr = pm.Data("cc_indptr", np.array([0, 1], dtype="int32"))
    # (cells, sets) one-hot of each cell's set, as CSR parts (one 1 per row).
    cs_indices = pm.Data("cs_indices", np.zeros(1, dtype="int32"))
    n_sets = interval_count.shape[0]
    n_cells = cs_indices.shape[0]
    cell_count = ps.CSR(cc_data, cc_indices, cc_indptr, pt.stack([n_cells, N_CELLS_TABLE]))
    cell_set_onehot = ps.CSR(
        pt.ones_like(cs_indices, dtype="float64"), cs_indices, pt.arange(n_cells + 1, dtype="int32"),
        pt.stack([n_cells, n_sets]),
    )
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))  # (cells, rules)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    # Each rule's prior log-weight is drawn around its family's level with a
    # stated spread RULE_SPREAD (about the incumbent's inferred tau), centred:
    # a fixed spread removes the tau-by-deviation funnel. Family levels sum to
    # zero, as the softmax over rules ignores a common shift.
    family_level = pm.ZeroSumNormal("family_level", sigma=1.5, shape=len(FAMILIES))
    rule_logw = pm.Normal("rule_logw", mu=family_level[RULE_FAMILY], sigma=RULE_SPREAD, shape=N_RULES)
    rule_weights = pm.Deterministic("rule_weights", pt.special.softmax(rule_logw))
    # Mean believed reliability and how certain people are of it.
    alpha_logit = pm.Normal("alpha_logit", mu=2.2, sigma=1.0)
    alpha = pm.Deterministic("alpha", pm.math.sigmoid(alpha_logit))
    log_kappa = pm.Normal("log_kappa", mu=np.log(15.0), sigma=0.75)
    kappa = pm.Deterministic("kappa", pt.exp(log_kappa))
    u, v = alpha * kappa, (1 - alpha) * kappa
    # Prior belief that the examples were sampled from the concept (strong
    # sampling) rather than merely produced consistently with it (weak).
    rho_logit = pm.Normal("rho_logit", mu=1.0, sigma=1.0)
    rho = pm.Deterministic("rho", pm.math.sigmoid(rho_logit))

    # Strong sampling: p(X | h) for every (n, k, s), reliability marginalised;
    # the Beta-function factor depends only on (n, j), so it is computed on that
    # small grid and broadcast over (k, s).
    beta_nj = pt.exp(_betaln(u + _J_NJ, v + _N_MINUS_J) - _betaln(u, v))  # (n, 1, 1, j)
    lik_strong = (_EXP_CONST * beta_nj).sum(axis=3)  # (n, k, s)
    # Weak sampling: every non-member must be a stray; size carries no evidence.
    lik_weak = _WEAK_CONST * pt.exp(_betaln(u, v + _NK) - _betaln(u, v))  # (n, k, s)
    lik_table = (rho * lik_strong + (1 - rho) * lik_weak).flatten()
    log_table = pt.log(lik_table)

    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + log_table[rule_flat_idx]

    int_ev = pt.dot(interval_count, lik_table)  # (sets,)
    log_intervals = pt.log1p(-lambda_0) - 2 * np.log(100) + pt.log(int_ev)

    log_joint = pt.concatenate([log_rules, log_intervals[:, None]], axis=1)
    log_norm = pt.logsumexp(log_joint, axis=1)
    rule_post = pt.exp(log_rules - log_norm[:, None])
    interval_post = pt.exp(log_intervals - log_norm)

    # Per (set, target) cell: posterior of each rule and of the intervals as a
    # group, and the intervals' evidence share that contains the target.
    post_by_set = pt.concatenate([rule_post, (interval_post / int_ev)[:, None]], axis=1)
    cell_post = ps.structured_dot(cell_set_onehot, post_by_set)  # (cells, rules + 1)
    cell_int_ev = ps.structured_dot(cell_count, lik_table[:, None])[:, 0]
    in_concept = (cell_post[:, :N_RULES] * cell_rule_member).sum(axis=1) + cell_post[:, N_RULES] * cell_int_ev
    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept, 1e-9, 1 - 1e-9)[row_cell])
    pm.Bernoulli("response", p=p_yes, observed=rating)


def _csr(prefix, dense):
    rows, cols = np.nonzero(dense)
    indptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=dense.shape[0]))])
    return {
        f"{prefix}_data": dense[rows, cols].astype("float64"),
        f"{prefix}_indices": cols.astype("int32"),
        f"{prefix}_indptr": indptr.astype("int32"),
    }


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    rule_k = np.zeros((S, N_RULES), dtype="int64")
    set_n_idx = np.zeros(S, dtype="int64")
    int_k = np.zeros((S, len(INT_LO)), dtype="int64")
    counts = np.zeros((S, MAX_N + 1, 100))
    for i, s in enumerate(unique):
        ex = np.array(s)
        set_n_idx[i] = len(ex) - 1
        rule_k[i] = RULE_MEMBERSHIP[:, ex - 1].sum(axis=1).astype("int64")
        int_k[i] = ((ex[None, :] >= INT_LO[:, None]) & (ex[None, :] <= INT_HI[:, None])).sum(axis=1)
        np.add.at(counts[i], (int_k[i], INT_SIZE_IDX), 1.0)
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: j for j, c in enumerate(cells)}
    cell_count = np.zeros((len(cells), MAX_N + 1, 100))
    for j, (i, t) in enumerate(cells):
        contains = (INT_LO <= t) & (t <= INT_HI)
        np.add.at(cell_count[j], (int_k[i][contains], INT_SIZE_IDX[contains]), 1.0)
    flat = lambda n_idx, c: np.concatenate(  # noqa: E731  place a (k, size) table in its n slot
        [c.ravel() if m == n_idx else np.zeros((MAX_N + 1) * 100) for m in range(MAX_N)]
    )
    cell_set = np.array([i for i, _ in cells], dtype="int64")
    return {
        "rule_flat_idx": (set_n_idx[:, None] * (MAX_N + 1) + rule_k) * 100 + RULE_SIZE_IDX[None, :],
        "interval_count": np.stack([flat(set_n_idx[i], counts[i]) for i in range(S)]),
        **_csr("cc", np.stack([flat(set_n_idx[i], cell_count[j]) for j, (i, _) in enumerate(cells)])),
        "cs_indices": cell_set.astype("int32"),
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]),
        "row_cell": np.array([cell_index[c] for c in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
