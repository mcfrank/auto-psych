"""Hundreds-chart block concepts.

People picture the numbers 1-100 laid out as a ten-by-ten hundreds chart (rows
are decades, columns are final digits), and the "magnitude" concepts they
entertain are rectangular blocks of that chart rather than stretches of the
number line. They weigh these blocks, together with familiar mathematical and
digit rules, by how small a concept can still cover the examples (a tempered
size principle), and say "yes" with their belief that the concept contains the
target. Examples sharing a final digit are read as a narrow column block that
excludes other final digits even of the same parity; when no rule fits,
in-range targets are endorsed only if their final digit lies within the span of
the examples' final digits.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
NUMBERS = np.arange(1, 101)
ROW = (NUMBERS - 1) // 10
COL = (NUMBERS - 1) % 10


def _is_prime(n):
    return n > 1 and all(n % a for a in range(2, int(n ** 0.5) + 1))


def rule_concepts():
    """45 familiar rules, each tagged with its family."""
    rules = [("odd", 0, [n for n in NUMBERS if n % 2]), ("even", 0, list(range(2, 101, 2)))]
    rules += [("primes", 1, [n for n in NUMBERS if _is_prime(n)]),
              ("squares", 1, [n * n for n in range(1, 11)]),
              ("cubes", 1, [n ** 3 for n in range(1, 5)]),
              ("pow2 not 32", 1, [2, 4, 8, 16, 64]),
              ("pow2 or 37", 1, [2, 4, 8, 16, 32, 37, 64])]
    rules += [(f"powers of {m}", 1, [m ** k for k in range(1, 8) if m ** k <= 100]) for m in range(2, 11)]
    rules += [(f"multiples of {m}", 2, list(range(m, 101, m))) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]]
    rules += [(f"ends in {d}", 3, [n for n in NUMBERS if n % 10 == d]) for d in range(10)]
    rules += [(f"contains {d}", 4, [n for n in NUMBERS if str(d) in str(n)]) for d in range(10)]
    return rules


_RULES = rule_concepts()
RULE_FAMILY = np.array([f for _, f, _ in _RULES], dtype="int64")
N_FAM = 5
FAM_COUNT = np.bincount(RULE_FAMILY, minlength=N_FAM).astype("float64")
RULE_MEM = np.array([np.isin(NUMBERS, c) for _, _, c in _RULES], dtype="float64")
RULE_SIZE = RULE_MEM.sum(axis=1)

# Every rectangular block of the hundreds chart.
_blocks = [(r1, r2, c1, c2) for r1 in range(10) for r2 in range(r1, 10)
           for c1 in range(10) for c2 in range(c1, 10)]
BLOCK_MEM = np.array([(ROW >= r1) & (ROW <= r2) & (COL >= c1) & (COL <= c2)
                      for r1, r2, c1, c2 in _blocks], dtype="float64")
BLOCK_SIZE = BLOCK_MEM.sum(axis=1).astype("int64")
SIZE_COUNT = np.bincount(BLOCK_SIZE, minlength=101).astype("float64")
SIZE_PRESENT = SIZE_COUNT > 0
LOG_SIZE_GRID = np.log(np.arange(101)[SIZE_PRESENT])
LOG_SIZE_COUNT = np.log(SIZE_COUNT[SIZE_PRESENT])


def _ll(n_in, size, n_ex):
    """Log-likelihood of the examples under a concept (10% strays from 1-100)."""
    return n_in * np.log(ALPHA / size + (1 - ALPHA) / 100) + (n_ex - n_in) * np.log((1 - ALPHA) / 100)


def _block_groups(examples):
    """Blocks sharing size and number of covered examples share prior and
    likelihood; summarise each group by log count, log size, log-lik, mean membership."""
    ex = np.array(examples) - 1
    n_in = BLOCK_MEM[:, ex].sum(axis=1).astype("int64")
    key = BLOCK_SIZE * 10 + n_in
    out = []
    for k in np.unique(key):
        sel = key == k
        s, m = int(BLOCK_SIZE[sel][0]), int(n_in[sel][0])
        out.append((np.log(sel.sum()), np.log(s), _ll(m, s, len(examples)), BLOCK_MEM[sel].mean(axis=0)))
    return out


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    n = len(unique)
    rule_ll = np.stack([_ll(RULE_MEM[:, np.array(s) - 1].sum(axis=1), RULE_SIZE, len(s)) for s in unique])
    groups = [_block_groups(s) for s in unique]
    g = max(len(x) for x in groups)
    b_logn = np.full((n, g), -1e4)
    b_logsize = np.zeros((n, g))
    b_ll = np.zeros((n, g))
    b_mem = np.zeros((n, g, 100))
    for i, gr in enumerate(groups):
        for j, (ln, ls, ll, mem) in enumerate(gr):
            b_logn[i, j], b_logsize[i, j], b_ll[i, j], b_mem[i, j] = ln, ls, ll, mem
    cell_keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(cell_keys))
    cell_index = {c: i for i, c in enumerate(cells)}
    cell_set = np.array([c[0] for c in cells], dtype="int64")
    cell_t = np.array([c[1] - 1 for c in cells], dtype="int64")
    cell_mem = np.concatenate([RULE_MEM[:, cell_t].T, b_mem[cell_set, :, cell_t]], axis=1)
    return {
        "cell_set": cell_set,
        "cell_membership": cell_mem,
        "rule_log_lik": rule_ll,
        "block_log_count": b_logn,
        "block_log_size": b_logsize,
        "block_log_lik": b_ll,
        "row_cell": np.array([cell_index[c] for c in cell_keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_membership = pm.Data("cell_membership", np.zeros((1, 2)))
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, len(_RULES))))
    block_log_count = pm.Data("block_log_count", np.zeros((1, 1)))
    block_log_size = pm.Data("block_log_size", np.zeros((1, 1)))
    block_log_lik = pm.Data("block_log_lik", np.zeros((1, 1)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Prior share of rules versus chart blocks, and of each rule family.
    lambda_0 = pm.Beta("lambda_0", 2.0, 2.0)
    family_weights = pm.Dirichlet("family_weights", a=np.full(N_FAM, 2.0))
    # Strength of the size principle (evidence tempering).
    log_beta = pm.Normal("log_beta", mu=0.0, sigma=0.75)
    beta = pt.exp(log_beta)
    # Block prior proportional to size ** size_power.
    size_power = pm.Normal("size_power", mu=0.0, sigma=1.0)

    log_rule_prior = pt.log(lambda_0) + pt.log(family_weights)[RULE_FAMILY] - np.log(FAM_COUNT)[RULE_FAMILY]
    log_z = pt.logsumexp(size_power * LOG_SIZE_GRID + LOG_SIZE_COUNT)
    log_block = (pt.log1p(-lambda_0) - log_z + size_power * block_log_size
                 + block_log_count + beta * block_log_lik)
    full = pt.concatenate([log_rule_prior[None, :] + beta * rule_log_lik, log_block], axis=1)
    posterior = pt.exp(full - pt.logsumexp(full, axis=1, keepdims=True))
    in_concept = (posterior[cell_set] * cell_membership).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)
