"""Working-memory-limited concept learning over a random subset of the examples.

People cannot hold every example in working memory while judging the target:
each example shown is retained independently with an inferred probability r, and
the concept is inferred only from the retained examples by the same conservative
Bayesian learner as `outlier_tolerant_rule_concepts` (42 rules, the 5,050
intervals of 1..100 with a Beta midpoint prior and decade-aligned bonus, size
principle with an inferred stray-example rate, evidence weight beta * k ** gamma
with k the number of examples held, guessing that falls with the number shown).
The answer averages over which non-empty subset was retained, with probability
r^k (1-r)^(n-k) / (1 - (1-r)^n). Order of presentation plays no role.

Computation (exact, organised to be cheap): an interval's weight for a retained
subset of size k depends on its prior and on (k, how many held examples it
contains), so only 15 weight grids over (lower, upper) edges are built. For each
subset, the mass of intervals with each upper edge (and each lower edge) is a
few segment sums of cumulative sums of those grids (segments precomputed in
prepare_observed); the mass containing a target is the total minus the mass of
intervals lying wholly below or wholly above it.
"""

import itertools

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_SET = 4
COMBOS = [(k, c) for k in range(1, MAX_SET + 1) for c in range(k + 1)]  # (held, contained)
COMBO_ID = {kc: i for i, kc in enumerate(COMBOS)}
N_COMBO = len(COMBOS)
COMBO_K = np.array([k for k, _ in COMBOS], dtype="float64")
COMBO_C = np.array([c for _, c in COMBOS], dtype="float64")
MAX_SEG = MAX_SET + 1


def concept_sets():
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


RULES = concept_sets()
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")  # (rules, 100)
LOG_RULE_SIZE = np.log(RULE_MEMBERSHIP.sum(axis=1))

# Interval grid [a, b] (0-based lower a, upper b), valid when a <= b.
LO, HI = np.meshgrid(NUMBERS, NUMBERS, indexing="ij")
VALID = (LO <= HI).astype("float64")
LOG_LEN = np.log(np.maximum(HI - LO + 1, 1).astype("float64"))
MIDPOINT_GRID = np.clip(np.arange(199) / 198, 1e-5, 1 - 1e-5)
GRID_POINT = LO + HI - 2
DECADE_STARTS = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90]
DECADE_ENDS = [9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 100]
DECADE_ALIGNED = (np.isin(LO, DECADE_STARTS) & np.isin(HI, DECADE_ENDS)).astype("float64") * VALID
BELOW = (NUMBERS[:, None] < NUMBERS[None, :]).astype("float64")  # [b, t]: edge b+1 < target t+1
ABOVE = BELOW.T  # [a, t]: edge a+1 > target t+1

with pm.Model() as model:
    sub_rule_n = pm.Data("sub_rule_n", np.zeros((1, N_RULES)))  # (subsets, rules): held examples r contains
    sub_held = pm.Data("sub_held", np.ones(1, dtype="float64"))  # (subsets,): examples held
    # Segment corners, flat indices into the cumulative grids: (subsets, 100 edges, MAX_SEG).
    col_hi = pm.Data("col_hi", np.zeros((1, 100, MAX_SEG), dtype="int64"))
    col_lo = pm.Data("col_lo", np.zeros((1, 100, MAX_SEG), dtype="int64"))
    row_hi = pm.Data("row_hi", np.zeros((1, 100, MAX_SEG), dtype="int64"))
    row_lo = pm.Data("row_lo", np.zeros((1, 100, MAX_SEG), dtype="int64"))
    set_sub_held = pm.Data("set_sub_held", np.ones((1, 1)))  # (sets, subsets): k if subset of set
    set_sub_lost = pm.Data("set_sub_lost", np.zeros((1, 1)))  # (sets, subsets): n - k forgotten
    set_sub_mask = pm.Data("set_sub_mask", np.ones((1, 1)))  # (sets, subsets): 1 if subset of set
    set_size = pm.Data("set_size", np.ones(1, dtype="float64"))  # (sets,)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_target = pm.Data("cell_target", np.zeros(1, dtype="int64"))  # target - 1
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 2, 2)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    midpoint_shape = pm.Gamma("midpoint_shape", alpha=2, beta=1, shape=2)
    evidence_strength = pm.LogNormal("evidence_strength", mu=0.0, sigma=0.5)
    size_growth = pm.Normal("size_growth", mu=0.0, sigma=1.0)
    decade_bonus = pm.Normal("decade_bonus", mu=0.0, sigma=3.0)

    # Each held example a concept contains gains log P(x | h) - log P(x | stray);
    # the stray term common to every concept cancels in the normalisation.
    stray_rate = pm.Beta("stray_rate", 1.5, 6.0)
    log_stray = pt.log(stray_rate / 100)
    rule_gain = pt.log((1 - stray_rate) * pt.exp(-LOG_RULE_SIZE) + stray_rate / 100) - log_stray  # (rules,)
    int_gain = pt.log((1 - stray_rate) * pt.exp(-LOG_LEN) + stray_rate / 100) - log_stray  # (100, 100)
    max_gain = pt.log((1 - stray_rate) + stray_rate / 100) - log_stray  # length-1 gain

    log_density = pm.logp(pm.Beta.dist(midpoint_shape[0], midpoint_shape[1]), MIDPOINT_GRID)
    log_int = log_density[GRID_POINT] + decade_bonus * DECADE_ALIGNED
    int_max = pt.max(log_int)
    log_int_prior = pt.log1p(-p_mix) + log_int - int_max - pt.log(pt.sum(pt.exp(log_int - int_max) * VALID))

    # Weight grids per (held k, contained c), sharing a per-k offset so subsets compare exactly.
    combo_w = evidence_strength * pt.exp(size_growth * pt.log(COMBO_K))  # (combos,)
    combo_off = 0.5 * combo_w * COMBO_K * max_gain
    log_grid = (log_int_prior[None] + (combo_w * COMBO_C)[:, None, None] * int_gain[None]
                - combo_off[:, None, None])
    grid = pt.exp(pt.clip(log_grid, -700.0, 700.0)) * VALID[None]  # (combos, a, b)
    zero_row = pt.zeros((N_COMBO, 1, 100))
    cum_a = pt.concatenate([zero_row, pt.cumsum(grid, axis=1)], axis=1).reshape((-1,))  # [c, a', b]: sum a < a'
    rev_b = pt.cumsum(grid[:, :, ::-1], axis=2)[:, :, ::-1]
    cum_b = pt.concatenate([rev_b, pt.zeros((N_COMBO, 100, 1))], axis=2).reshape((-1,))  # [c, a, b']: sum b >= b'

    upper_mass = pt.sum(cum_a[col_hi] - cum_a[col_lo], axis=2)  # (subsets, b): intervals with upper edge b
    lower_mass = pt.sum(cum_b[row_lo] - cum_b[row_hi], axis=2)  # (subsets, a): intervals with lower edge a
    int_total = pt.sum(upper_mass, axis=1)  # (subsets,)
    int_in = int_total[:, None] - pt.dot(upper_mass, BELOW) - pt.dot(lower_mass, ABOVE)  # (subsets, 100)

    sub_w = evidence_strength * pt.exp(size_growth * pt.log(sub_held))  # (subsets,)
    sub_off = 0.5 * sub_w * sub_held * max_gain
    log_rule = (pt.log(p_mix) + pt.log(rule_weights)[None, :] + sub_w[:, None] * sub_rule_n * rule_gain[None, :]
                - sub_off[:, None])
    rule_w = pt.exp(pt.clip(log_rule, -700.0, 700.0))  # (subsets, rules)
    total = pt.sum(rule_w, axis=1) + int_total + 1e-300
    sub_in_concept = pt.clip((pt.dot(rule_w, RULE_MEMBERSHIP) + int_in) / total[:, None], 0.0, 1.0)

    # The one change: each shown example is retained with probability `retention`.
    retention = pm.Beta("retention", 4.0, 1.5)
    sub_weight = set_sub_mask * pt.exp(set_sub_held * pt.log(retention) + set_sub_lost * pt.log1p(-retention))
    sub_weight = sub_weight / pt.sum(sub_weight, axis=1, keepdims=True)  # (sets, subsets)
    in_concept = pt.dot(sub_weight, sub_in_concept)[cell_set, cell_target]  # (cells,)

    guess_single = pm.Normal("guess_single", mu=-2.0, sigma=1.5)
    guess_decline = pm.Normal("guess_decline", mu=0.0, sigma=2.0)
    guess_yes = pm.Beta("guess_yes", 2, 2)
    guess_rate = pm.math.sigmoid(guess_single - guess_decline * pt.log(set_size))[cell_set]  # (cells,)
    q = guess_rate * guess_yes + (1 - guess_rate) * in_concept
    p_yes = pm.Deterministic("p_yes", pt.clip(q[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def _segments(counts):
    """Runs of equal value in `counts` as (value, start, end) with end inclusive."""
    out, start = [], 0
    for i in range(1, len(counts) + 1):
        if i == len(counts) or counts[i] != counts[start]:
            out.append((int(counts[start]), start, i - 1))
            start = i
    return out


def _corners(sub):
    """Flat cumulative-grid indices for the per-edge segment sums of one held subset."""
    k, x = len(sub), np.array(sub) - 1
    shape = (MAX_SEG,)
    col_hi = np.zeros((100,) + shape, dtype="int64")
    col_lo = np.zeros_like(col_hi)
    row_hi = np.zeros_like(col_hi)
    row_lo = np.zeros_like(col_hi)
    for e in range(100):
        # Column b = e: lower edges a = 0..e, contained count = #x in [a, e]; cum_a is (combos, 101, 100).
        cnt = ((x[None, :] >= np.arange(e + 1)[:, None]) & (x[None, :] <= e)).sum(axis=1)
        for j, (c, a0, a1) in enumerate(_segments(cnt)):
            base = COMBO_ID[(k, c)] * 101 * 100
            col_hi[e, j] = base + (a1 + 1) * 100 + e
            col_lo[e, j] = base + a0 * 100 + e
        # Row a = e: upper edges b = e..99, contained count = #x in [e, b]; cum_b is (combos, 100, 101).
        bs = np.arange(e, 100)
        cnt = ((x[None, :] >= e) & (x[None, :] <= bs[:, None])).sum(axis=1)
        for j, (c, s0, s1) in enumerate(_segments(cnt)):
            base = COMBO_ID[(k, c)] * 100 * 101
            row_lo[e, j] = base + e * 101 + (e + s0)
            row_hi[e, j] = base + e * 101 + (e + s1 + 1)
    return col_hi, col_lo, row_hi, row_lo


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    cells = list(dict.fromkeys(zip(sets, targets)))
    cell_index = {c: i for i, c in enumerate(cells)}
    subsets = list(dict.fromkeys(
        sub for s in unique for k in range(1, len(s) + 1) for sub in itertools.combinations(sorted(set(s)), k)
    ))
    sub_index = {sub: j for j, sub in enumerate(subsets)}
    mask = np.zeros((len(unique), len(subsets)))
    held = np.zeros_like(mask)
    lost = np.zeros_like(mask)
    for i, s in enumerate(unique):
        u = sorted(set(s))
        for k in range(1, len(u) + 1):
            for sub in itertools.combinations(u, k):
                j = sub_index[sub]
                mask[i, j], held[i, j], lost[i, j] = 1.0, float(k), float(len(u) - k)
    corners = [_corners(sub) for sub in subsets]
    return {
        "sub_rule_n": np.stack([RULE_MEMBERSHIP[:, np.array(sub) - 1].sum(axis=1) for sub in subsets]),
        "sub_held": np.array([float(len(sub)) for sub in subsets], dtype="float64"),
        "col_hi": np.stack([c[0] for c in corners]),
        "col_lo": np.stack([c[1] for c in corners]),
        "row_hi": np.stack([c[2] for c in corners]),
        "row_lo": np.stack([c[3] for c in corners]),
        "set_sub_held": held,
        "set_sub_lost": lost,
        "set_sub_mask": mask,
        "set_size": np.array([float(len(s)) for s in unique], dtype="float64"),
        "cell_set": np.array([index[s] for s, _ in cells], dtype="int64"),
        "cell_target": np.array([t - 1 for _, t in cells], dtype="int64"),
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
