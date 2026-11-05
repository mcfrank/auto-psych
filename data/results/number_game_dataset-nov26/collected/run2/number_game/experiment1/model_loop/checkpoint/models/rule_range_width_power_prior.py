"""Rule-within-range conjunction with a width-power prior on the range.

Refinement of rule_within_range_conjunction: people infer a conjunctive
concept, a mathematical rule (or "any number") restricted to a range of
magnitudes, inferring both by Bayesian size-principle reasoning, and say "yes"
with the posterior probability that the concept contains the target. The one
change: a restricting range [m, n] has prior weight proportional to its width
(n - m + 1) raised to an inferred exponent beta (beta = 0 recovers the
incumbent's uniform range prior), so a preference for narrow or broad ranges
governs how far a rule is extended beyond the examples' span.

Intervals of equal width share prior weight, so for every set, rule and width
the log evidence summed over intervals of that width and, per observed
target, the within-width probability the interval covers the target are
precomputed once. For speed the 100 widths are pooled into 24 log-spaced
width bands: each band's total prior mass is exact (sum over its widths of
count * width**beta), and within a band intervals are weighted equally.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

ALPHA = 0.9
NUMBERS = np.arange(1, 101)
WIDTHS = np.arange(1, 101)
LOG_W = np.log(WIDTHS.astype("float64"))
LOG_COUNT = np.log((101 - WIDTHS).astype("float64"))  # number of intervals of each width
_EDGES = np.unique(np.round(np.geomspace(1, 101, 25)).astype(int))
BAND = np.searchsorted(_EDGES, WIDTHS, side="right") - 1  # band of each width
N_B = int(BAND.max()) + 1
BAND_MAT = np.array([BAND == b for b in range(N_B)], dtype="float64")  # (B, W)
LOG_BAND_COUNT = np.log(BAND_MAT @ (101 - WIDTHS))


def rule_concepts():
    rules = {
        "any number": list(NUMBERS),
        "odd": [n for n in NUMBERS if n % 2 == 1],
        "even": list(range(2, 101, 2)),
        "primes": [n for n in NUMBERS if n > 1 and all(n % a for a in range(2, int(n**0.5) + 1))],
        "squares": [n * n for n in range(1, 11)],
        "cubes": [n**3 for n in range(1, 5)],
        "powers of 2, not 32": [2, 4, 8, 16, 64],
        "powers of 2, or 37": [2, 4, 8, 16, 32, 37, 64],
    }
    rules.update({f"multiples of {m}": list(range(m, 101, m)) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]})
    rules.update({f"powers of {m}": [m**k for k in range(1, 8) if m**k <= 100] for m in range(2, 11)})
    rules.update({f"ends in {d}": [n for n in NUMBERS if n % 10 == d] for d in range(10)})
    rules.update({f"contains digit {d}": [n for n in NUMBERS if str(d) in str(n)] for d in range(1, 10)})
    return rules


RULES = list(rule_concepts().values())
N_RULES = len(RULES)
RULE_MEM = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")  # (R, 100)
_M, _N = np.meshgrid(np.arange(100), np.arange(100), indexing="ij")
# Skewed index: interval of width w (index w-1) starting at m ends at m + w - 1.
_WI, _MI = np.meshgrid(np.arange(100), np.arange(100), indexing="ij")
_END = _MI + _WI
_OK = _END <= 99
_ENDC = np.minimum(_END, 99)
_T = np.arange(100)


def set_tables(examples):
    """Returns log evidence of the unrestricted rule (R,), log evidence summed
    over intervals of each width (R, W), and within-width coverage (R, W, 100)."""
    ex = np.array(examples) - 1
    log_full = np.zeros(N_RULES)
    width_ev = np.zeros((N_RULES, 100))
    cover = np.zeros((N_RULES, 100, 100))
    hi_m = np.minimum(_T[None, :], 99 - np.arange(100)[:, None])  # (W, T) last start covering t
    lo_m = np.maximum(0, _T[None, :] - np.arange(100)[:, None])  # (W, T) first start covering t
    for r in range(N_RULES):
        mem = RULE_MEM[r]
        csum = np.concatenate([[0.0], np.cumsum(mem)])
        size = csum[_N + 1] - csum[_M]
        ll = np.zeros((100, 100))
        for x in ex:
            inside = (mem[x] > 0) & (_M <= x) & (x <= _N)
            pin = np.where(inside & (size > 0), ALPHA / np.maximum(size, 1), 0.0)
            ll += np.log(pin + (1 - ALPHA) / 100)
        log_full[r] = ll[0, 99]
        L = np.where(_OK, ll[_MI, _ENDC], -np.inf)  # (W, M)
        E = logsumexp(L, axis=1)  # (W,)
        width_ev[r] = E
        p = np.exp(L - E[:, None])
        cs = np.concatenate([np.zeros((100, 1)), np.cumsum(p, axis=1)], axis=1)  # (W, 101)
        rows = np.arange(100)[:, None]
        cov = np.where(hi_m >= lo_m, cs[rows, hi_m + 1] - cs[rows, lo_m], 0.0)
        cover[r] = np.clip(cov, 0.0, 1.0) * mem[None, :]
    return log_full, width_ev, cover


def band_tables(examples):
    """Per rule: unrestricted log evidence (R,), log mean evidence over the
    intervals of each width band (R, B), and band coverage (R, B, 100)."""
    log_full, width_ev, cover = set_tables(examples)
    band_ev = np.stack([logsumexp(width_ev[:, BAND == b], axis=1) for b in range(N_B)], axis=1)
    band_cov = np.zeros((N_RULES, N_B, 100))
    for b in range(N_B):
        sel = BAND == b
        wts = np.exp(width_ev[:, sel] - band_ev[:, b : b + 1])  # (R, w in band)
        band_cov[:, b] = (wts[:, :, None] * cover[:, sel]).sum(axis=1)
    return log_full, band_ev - LOG_BAND_COUNT[None, :], np.clip(band_cov, 0.0, 1.0)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) - 1 for r in rows]
    cells = list(dict.fromkeys(zip(sets, targets)))
    cidx = {c: i for i, c in enumerate(cells)}
    unique = list(dict.fromkeys(sets))
    sidx = {s: i for i, s in enumerate(unique)}
    tables = {s: band_tables(s) for s in unique}
    onehot = np.zeros((len(cells), len(unique)))
    onehot[np.arange(len(cells)), [sidx[s] for s, t in cells]] = 1.0
    return {
        "full_log_lik": np.stack([tables[s][0] for s in unique]),  # (S, R)
        "band_log_ev": np.stack([tables[s][1] for s in unique]),  # (S, R, B)
        "cell_set_onehot": onehot,  # (C, S)
        "cell_cover": np.ascontiguousarray(np.stack([tables[s][2][:, :, t] for s, t in cells])),
        "cell_full_mem": np.stack([RULE_MEM[:, t] for s, t in cells]),  # (C, R)
        "row_cell": np.array([cidx[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    full_log_lik = pm.Data("full_log_lik", np.zeros((1, N_RULES)))
    band_log_ev = pm.Data("band_log_ev", np.zeros((1, N_RULES, N_B)))
    cell_cover = pm.Data("cell_cover", np.zeros((1, N_RULES, N_B)))
    cell_full_mem = pm.Data("cell_full_mem", np.zeros((1, N_RULES)))
    cell_set_onehot = pm.Data("cell_set_onehot", np.zeros((1, 1)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    q_full = pm.Beta("q_full", 2, 2)  # probability the rule is applied over all of 1..100
    beta = pm.Normal("beta", 0.0, 1.5)  # width exponent of the range prior

    # prior mass of each width band (exact), normalised over all intervals
    log_mass_w = beta * pt.as_tensor(LOG_W) + pt.as_tensor(LOG_COUNT)  # (W,)
    log_band = pt.log(pt.dot(pt.as_tensor(BAND_MAT), pt.exp(log_mass_w - pt.max(log_mass_w))))
    log_band = log_band - pt.logsumexp(log_band)  # (B,)
    lb_joint = log_band[None, None, :] + band_log_ev  # (S, R, B)
    log_sub = pt.logsumexp(lb_joint, axis=2)  # (S, R)

    log_w = pt.log(rule_weights)
    log_a = log_w[None, :] + pt.log(q_full) + full_log_lik  # (S, R)
    log_b = log_w[None, :] + pt.log1p(-q_full) + log_sub  # (S, R)
    log_norm = pt.logsumexp(pt.concatenate([log_a, log_b], axis=1), axis=1, keepdims=True)
    post_full = pt.exp(log_a - log_norm)  # (S, R)
    post_rb = pt.exp(lb_joint + (log_b - log_sub - log_norm)[:, :, None])  # (S, R, B)

    # route each set's posterior to its (set, target) cells
    post_full_c = pt.dot(cell_set_onehot, post_full)  # (C, R)
    post_rb_c = pt.dot(cell_set_onehot, post_rb.reshape((post_rb.shape[0], -1)))  # (C, R*B)
    in_cell = (post_full_c * cell_full_mem).sum(axis=1) + (
        post_rb_c * cell_cover.reshape((cell_cover.shape[0], -1))
    ).sum(axis=1)
    p_yes = pm.Deterministic("p_yes", pt.clip(in_cell[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)
