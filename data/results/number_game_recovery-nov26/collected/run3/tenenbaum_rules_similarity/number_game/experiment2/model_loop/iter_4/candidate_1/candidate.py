"""Directional range momentum: people read the examples in the order shown as a
path along the number line, and when they consider a magnitude range they expect
it to continue in the direction the examples moved (first example toward last),
extending further ahead of the path than behind it, to an inferred degree tau.
Otherwise this is the incumbent mixed_strong_weak_sampling_concepts learner: 45
rules with a family-organised prior (exception rules their own family), all
intervals of 1..100 with a beta prior over the share of the line a range covers,
mixed strong/weak sampling with 10% stray examples. Within each range width, the
placement prior is tilted towards intervals whose centre lies ahead of the
examples' midpoint in the direction of travel (normalised within width, so the
rule-vs-range balance and the width prior are untouched). tau = 0 recovers the
incumbent; sets of one example (or first == last) carry no direction.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
NUMBERS = np.arange(1, 101)


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
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
SIZES = np.arange(1, 101, dtype="float64")
SHARE = (SIZES - 0.5) / 100.0  # share of the 1..100 line covered, in (0, 1)

DIRECTIONS = np.array([-1.0, 0.0, 1.0])  # descending, none, ascending presentation
A_GRID = np.arange(101, dtype="float64")  # start positions 0..100 (0 = empty prefix)


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


FAMILY = np.array([rule_family(n) for n in RULE_NAMES])
N_FAMILIES = FAMILY.max() + 1
FAMILY_SIZE = np.bincount(FAMILY)


def member_log_lik(theta, size):
    """Log probability of one example that the concept contains (mixed sampling + strays)."""
    return pt.log(ALPHA * (theta / size + (1.0 - theta) / 100.0) + (1.0 - ALPHA) / 100.0)


LOG_STRAY = np.log((1.0 - ALPHA) / 100.0)

# For one set and one width, consecutive placements holding the same number of
# examples form a segment; prepare_observed lists the segments of every width
# (parameter-free). A segment's prior mass is a geometric sum over its starts,
# taken from the prefix sums Psum[d, a] = sum_{a' <= a} exp(tau * d * a' / 100).

with pm.Model() as model:
    set_n = pm.Data("set_n", np.zeros(1))  # (sets,) number of examples
    rule_k = pm.Data("rule_k", np.zeros((1, N_RULES)))  # (sets, rules) examples inside
    seg_w = pm.Data("seg_w", np.zeros((1, 1), dtype="int64"))  # (sets, segs) width index s = w - 1
    seg_k = pm.Data("seg_k", np.zeros((1, 1)))  # examples inside each interval of the segment
    seg_hi = pm.Data("seg_hi", np.zeros((1, 1), dtype="int64"))  # flat Psum index of last start
    seg_lo = pm.Data("seg_lo", np.zeros((1, 1), dtype="int64"))  # flat Psum index of first start - 1
    seg_norm = pm.Data("seg_norm", np.zeros((1, 1), dtype="int64"))  # flat Psum index of 100 - s
    seg_pad = pm.Data("seg_pad", np.zeros((1, 1)))  # 1 for padding entries
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_target = pm.Data("cell_target", np.zeros(1, dtype="int64"))  # 0-based
    q_seg = pm.Data("q_seg", np.zeros((1, 1), dtype="int64"))  # (cells, 2*widths) flat segment index
    q_p = pm.Data("q_p", np.zeros((1, 1), dtype="int64"))  # flat Psum index of the query start
    q_sign = pm.Data("q_sign", np.zeros((1, 1)))  # +1 upper end, -1 lower end
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    mean_share = pm.Beta("mean_share", 2, 3)
    concentration = pm.LogNormal("concentration", mu=np.log(5.0), sigma=1.0)
    family_weights = pm.Dirichlet("family_weights", a=np.full(N_FAMILIES, 2.0))
    theta = pm.Beta("theta", 4.0, 1.0)  # share of strong (size-principle) sampling
    tau = pm.Normal("tau", mu=0.0, sigma=3.0)  # momentum: tilt toward ranges ahead of the path
    rule_log_prior = pt.log(family_weights)[FAMILY] - np.log(FAMILY_SIZE[FAMILY])

    log_size_prior = ((mean_share * concentration - 1.0) * np.log(SHARE)
                      + ((1.0 - mean_share) * concentration - 1.0) * np.log1p(-SHARE))
    log_size_prior = log_size_prior - pt.logsumexp(log_size_prior)

    # placement weight exp(tau * direction * (centre - examples' midpoint) / 100); within a
    # width only the start position survives normalisation (centred at 50 for stability)
    step = pt.exp(tau * DIRECTIONS[:, None] * (A_GRID[None, :] - 50.0) / 100.0)
    step = pt.set_subtensor(step[:, 0], 0.0)
    psum = pt.cumsum(step, axis=1).reshape((-1,))  # (3 * 101,)
    seg_mass = psum[seg_hi] - psum[seg_lo]
    log_place = pt.log(seg_mass) - pt.log(psum[seg_norm])

    rule_ll = (rule_k * member_log_lik(theta, RULE_SIZE)[None, :]
               + (set_n[:, None] - rule_k) * LOG_STRAY)
    seg_ll = seg_k * member_log_lik(theta, SIZES)[seg_w] + (set_n[:, None] - seg_k) * LOG_STRAY

    log_rules = pt.log(lambda_0) + rule_log_prior[None, :] + rule_ll
    log_segs = pt.log1p(-lambda_0) + log_size_prior[seg_w] + log_place + seg_ll - 1e4 * seg_pad
    log_joint = pt.concatenate([log_rules, log_segs], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    seg_post = posterior[:, N_RULES:]
    # cumulative range mass up to a start a inside segment j: mass of earlier segments
    # plus the share of segment j's mass from starts <= a
    ratio = seg_post / seg_mass
    offset = (pt.cumsum(seg_post, axis=1) - seg_post) - ratio * psum[seg_lo]
    in_range = (q_sign * (offset.reshape((-1,))[q_seg] + ratio.reshape((-1,))[q_seg] * psum[q_p])).sum(axis=1)
    in_rule = (posterior[cell_set, :N_RULES] * pt.as_tensor_variable(RULE_MEMBERSHIP.T)[cell_target]).sum(axis=1)
    in_concept = in_rule + in_range

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def set_segments(ex):
    """Segments (width index, first start, last start, examples inside) of one set."""
    segs = []
    for s in range(100):
        a = np.arange(1, 101 - s)
        k = sum(((a <= e) & (e <= a + s)).astype(int) for e in ex)
        breaks = np.flatnonzero(np.diff(k)) + 1
        for lo, hi in zip(np.r_[0, breaks], np.r_[breaks, len(a)]):
            segs.append((s, int(a[lo]), int(a[hi - 1]), int(k[lo])))
    return segs


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    all_segs = [set_segments(np.array(s)) for s in unique]
    n_seg = max(len(g) for g in all_segs)
    shape = (len(unique), n_seg)
    seg_w, seg_k, seg_hi, seg_lo, seg_norm = (np.zeros(shape, dtype="int64") for _ in range(5))
    seg_pad = np.ones(shape)
    seg_a1 = np.ones(shape, dtype="int64")
    direction = np.array([int(np.sign(s[-1] - s[0])) + 1 for s in unique])
    lookup = []  # per set: (width, start) -> flat segment index
    for i, g in enumerate(all_segs):
        d = direction[i] * 101
        seg_hi[i], seg_lo[i], seg_norm[i] = d + 1, d, d + 100  # padding: harmless positive mass
        table = np.zeros((100, 101), dtype="int64")
        for j, (s, a1, a2, k) in enumerate(g):
            seg_w[i, j], seg_k[i, j], seg_pad[i, j], seg_a1[i, j] = s, k, 0.0, a1
            seg_hi[i, j], seg_lo[i, j], seg_norm[i, j] = d + a2, d + a1 - 1, d + 100 - s
            table[s, a1 - 1 if a1 == 1 else a1:a2 + 1] = i * n_seg + j
        lookup.append(table)
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: i for i, c in enumerate(cells)}
    widths = np.arange(100)
    q_seg = np.zeros((len(cells), 200), dtype="int64")
    q_p = np.zeros((len(cells), 200), dtype="int64")
    q_sign = np.r_[np.ones(100), -np.ones(100)][None, :].repeat(len(cells), axis=0)
    for c, (si, t) in enumerate(cells):
        hi = np.minimum(t, 100 - widths)  # intervals of width s holding t start in (lo, hi]
        lo = np.maximum(1, t - widths) - 1
        q_seg[c, :100] = lookup[si][widths, hi]
        q_seg[c, 100:] = lookup[si][widths, lo]
        q_p[c] = direction[si] * 101 + np.r_[hi, lo]
    rule_k = np.stack([RULE_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique])
    return {
        "set_n": np.array([len(s) for s in unique], dtype="float64"),
        "rule_k": rule_k,
        "seg_w": seg_w,
        "seg_k": seg_k.astype("float64"),
        "seg_hi": seg_hi,
        "seg_lo": seg_lo,
        "seg_norm": seg_norm,
        "seg_pad": seg_pad,
        "cell_set": np.array([c[0] for c in cells], dtype="int64"),
        "cell_target": np.array([c[1] - 1 for c in cells], dtype="int64"),
        "q_seg": q_seg,
        "q_p": q_p,
        "q_sign": q_sign,
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
