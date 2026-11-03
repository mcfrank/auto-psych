"""People infer arithmetic and final-digit rules (and stretches of them) by the
same Bayesian concept learning as no_digit_containment_span_concepts (rules
plus multiples of 13-50, span-restricted runs of the base rules, an inferred
share of random examples, rule examples drawn without replacement), but a
"region of magnitudes" is not a sharp-edged interval: it is a fuzzy cluster
around a central number (a discretised Gaussian over 1..100) from which the
program emits numbers more often the closer they are to the centre, and a
target's membership is its typicality, fading smoothly with distance from the
centre. Clusters replace the intervals; their widths are fixed multiples of
one inferred typical width.
So for examples fitting no rule "yes" peaks near the middle of the examples
and fades gradually beyond them, with no sharp cut-off at the span's edges.
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
    return rules


_RULES = rule_concepts()
N_BASE = len(_RULES)
BASE_MEMBERS = [sorted(c) for c in _RULES.values()]
EXT_M = list(range(13, 51))
N_EXT = len(EXT_M)
_RULES.update({f"multiples of {m}": list(range(m, 101, m)) for m in EXT_M})
RULES = list(_RULES.values())
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)

# Span-restricted rules: runs of 2..size-1 consecutive members of a base rule,
# every run equally likely a priori, grouped by (run length, examples inside).
MAX_RUN = max(len(m) for m in BASE_MEMBERS) - 1
SG_SIZE = np.repeat(np.arange(2, MAX_RUN + 1), MAX_EX + 1).astype("float64")
SG_NIN = np.tile(np.arange(MAX_EX + 1), MAX_RUN - 1).astype("float64")
N_SG = len(SG_SIZE)
N_RUNS_TOTAL = float(sum(len(m) - s + 1 for m in BASE_MEMBERS for s in range(2, len(m))))

# Gaussian magnitude clusters: every centre 1..100 crossed with a few widths,
# each a fixed multiple of one inferred typical width (all equally likely).
WIDTH_MULT = np.array([0.5, 1.0, 2.0, 4.0])
N_W = len(WIDTH_MULT)
N_G = 100 * N_W
CELL_C = np.repeat(np.arange(1, 101, dtype="float64"), N_W)
CELL_M = np.tile(WIDTH_MULT, 100)
SQ_DIST = (NUMBERS[None, :] - CELL_C[:, None]) ** 2  # (N_G, 100)
RULE_CONCENTRATION = np.full(N_BASE, 2.0)


def _wor_loglik(size, n_in, n_examples, noise):
    """Without-replacement log-likelihood for concepts of given size with n_in examples inside."""
    log_out = pt.log(noise / 100.0)
    ll = (n_examples[:, None] - n_in[None, :]) * log_out
    for j in range(MAX_EX):
        term = pt.log((1 - noise) / np.maximum(size - j, 1.0) + noise / 100.0)
        ll = ll + ((n_in > j) * term)[None, :]
    return ll


with pm.Model() as model:
    rule_n_in = pm.Data("rule_n_in", np.zeros((1, N_RULES)))
    n_examples = pm.Data("n_examples", np.ones(1))
    ex_counts = pm.Data("ex_counts", np.zeros((1, 100)))
    span_log_count = pm.Data("span_log_count", np.zeros((1, N_SG)))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_target = pm.Data("cell_target", np.zeros(1, dtype="int64"))
    cell_span_frac = pm.Data("cell_span_frac", np.zeros((1, N_SG)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    noise = pm.Beta("noise", 1.5, 6.0)
    rule_weights = pm.Dirichlet("rule_weights", a=RULE_CONCENTRATION)
    ext_share = pm.Beta("ext_share", 1.0, 4.0)
    span_share = pm.Beta("span_share", 1.0, 4.0)
    all_weights = pt.concatenate([
        (1 - ext_share) * rule_weights, pt.ones(N_EXT) * ext_share / N_EXT,
    ])
    log_span_prior = pt.log(lambda_0) + pt.log(span_share) - np.log(N_RUNS_TOTAL)

    # Typical cluster width (in numbers); a cluster's output distribution is a
    # discretised Gaussian, its typicality 1 at the centre.
    width = pm.LogNormal("width", mu=np.log(6.0), sigma=0.7)
    cluster_typ = pt.exp(-0.5 * SQ_DIST / (width * CELL_M[:, None]) ** 2)
    cluster_pmf = cluster_typ / cluster_typ.sum(axis=1, keepdims=True)
    log_per_cluster = -np.log(N_G)

    log_out = pt.log(noise / 100.0)
    rule_log_lik = (n_examples[:, None] - rule_n_in) * log_out
    for j in range(MAX_EX):
        rule_term = pt.log((1 - noise) / np.maximum(RULE_SIZE - j, 1.0) + noise / 100.0)
        rule_log_lik = rule_log_lik + pt.gt(rule_n_in, j) * rule_term[None, :]
    cluster_log_lik = pt.dot(ex_counts, pt.log((1 - noise) * cluster_pmf + noise / 100.0).T)
    span_log_lik = _wor_loglik(SG_SIZE, SG_NIN, n_examples, noise)

    log_groups = pt.log1p(-lambda_0) + log_per_cluster + cluster_log_lik
    log_rules = pt.log(lambda_0) + pt.log1p(-span_share) + pt.log(all_weights)[None, :] + rule_log_lik
    log_spans = log_span_prior + span_log_count + span_log_lik
    log_joint = pt.concatenate([log_rules, log_groups, log_spans], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    # Per set, the probability each of 1..100 belongs to a rule or a cluster.
    in_rule_cluster = (
        pt.dot(posterior[:, :N_RULES], RULE_MEMBERSHIP)
        + pt.dot(posterior[:, N_RULES:N_RULES + N_G], cluster_typ)
    )
    in_concept = in_rule_cluster[cell_set, cell_target] + (
        posterior[cell_set, N_RULES + N_G:] * cell_span_frac
    ).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-9, 1 - 1e-9))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set`, `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    rule_n_in = np.zeros((S, N_RULES))
    n_ex = np.zeros(S)
    ex_counts = np.zeros((S, 100))
    scounts = np.zeros((S, N_SG))
    scontain = np.zeros((S, N_SG, 100))
    for i, s in enumerate(unique):
        ex = np.array(s) - 1
        n_ex[i] = len(s)
        rule_n_in[i] = RULE_MEMBERSHIP[:, ex].sum(axis=1)
        np.add.at(ex_counts[i], ex, 1.0)
        for mem in BASE_MEMBERS:
            L = len(mem)
            is_ex = np.array([sum(1 for x in s if x == v) for v in mem])
            cum = np.concatenate([[0], np.cumsum(is_ex)])
            for size in range(2, L):
                base = (size - 2) * (MAX_EX + 1)
                for a in range(0, L - size + 1):
                    k = int(min(cum[a + size] - cum[a], MAX_EX))
                    g = base + k
                    scounts[i, g] += 1
                    scontain[i, g, np.array(mem[a:a + size]) - 1] += 1
    sfrac = np.where(scounts[:, :, None] > 0, scontain / np.maximum(scounts, 1)[:, :, None], 0.0)
    slog_count = np.where(scounts > 0, np.log(np.maximum(scounts, 1)), -1e3)
    keys = [(index[s], int(r["target"]) - 1) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {k: i for i, k in enumerate(cells)}
    cs = np.array([k[0] for k in cells], dtype="int64")
    ct = np.array([k[1] for k in cells], dtype="int64")
    return {
        "rule_n_in": rule_n_in,
        "n_examples": n_ex,
        "ex_counts": ex_counts,
        "span_log_count": slog_count,
        "cell_set": cs,
        "cell_target": ct,
        "cell_span_frac": sfrac[cs, :, ct],
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
