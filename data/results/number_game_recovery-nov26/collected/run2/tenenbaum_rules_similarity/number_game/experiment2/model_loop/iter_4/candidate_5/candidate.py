"""I chose large_number_biased_sampling and refined it. People do the same Bayesian
concept learning (45 independently weighted rules plus multiples of 13-50, intervals
of 1..100 with an Erlang width prior, an inferred share of random examples, examples
drawn without replacement), and they believe that when the program draws from a range
of magnitudes it favours the larger members (an in-range output n has weight n**gamma,
gamma inferred, sampled sequentially without replacement), but that rule concepts are
sampled uniformly. The one change is that the magnitude bias applies to ranges only:
a range stretching below the smallest example is cheap under the size principle, so
"yes" extends below examples that fit no rule more readily than above them, while
generalisation within rules is unchanged. "Yes" is the posterior probability that the
concept contains the target.
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
    rules.update({f"contains digit {d}": [n for n in NUMBERS if str(d) in str(n)] for d in range(10)})
    return rules


_RULES = rule_concepts()
N_BASE = len(_RULES)
EXT_M = list(range(13, 51))  # less-familiar multiples rules
N_EXT = len(EXT_M)
_RULES.update({f"multiples of {m}": list(range(m, 101, m)) for m in EXT_M})
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")  # (R, 100)
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)

# Every interval [lo, hi] of 1..100.
INT_LO, INT_HI = np.array([(a, b) for a in range(1, 101) for b in range(a, 101)]).T
N_INT = len(INT_LO)
INT_SIZE = (INT_HI - INT_LO + 1).astype("float64")
INT_MEMBERSHIP = ((NUMBERS[None, :] >= INT_LO[:, None]) & (NUMBERS[None, :] <= INT_HI[:, None])).astype("float64")
CONCEPT_MEMBERSHIP = np.concatenate([RULE_MEMBERSHIP, INT_MEMBERSHIP], axis=0)  # (C, 100)
SIZES_F = np.arange(1, 101).astype("float64")
N_PER_SIZE = 101.0 - SIZES_F
LOG_N = np.log(NUMBERS / 50.0)

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_BASE - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

N_CONC = N_RULES + N_INT
SUBSETS = np.array([[(b >> k) & 1 for k in range(MAX_EX)] for b in range(2**MAX_EX)], dtype="float64")

with pm.Model() as model:
    # One entry per (set, example, concept containing that example), built once per set.
    n_examples = pm.Data("n_examples", np.ones(1))  # (sets,)
    e_flat = pm.Data("e_flat", np.zeros(1, dtype="int64"))  # set * N_CONC + concept
    e_int = pm.Data("e_int", np.zeros(1, dtype="int64"))  # interval index (0 for rule entries)
    e_is_int = pm.Data("e_is_int", np.zeros(1))  # 1 if the concept is an interval
    e_rule_left = pm.Data("e_rule_left", np.ones(1))  # rule size minus earlier in-rule examples
    e_x = pm.Data("e_x", np.zeros(1, dtype="int64"))  # example value - 1
    set_ex = pm.Data("set_ex", np.full((1, MAX_EX), 100, dtype="int64"))  # (sets, 4) example - 1, 100 = none
    e_prev = pm.Data("e_prev", np.zeros(1, dtype="int64"))  # set * 16 + subset of earlier in-concept examples
    row_set = pm.Data("row_set", np.zeros(1, dtype="int64"))
    row_target = pm.Data("row_target", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    noise = pm.Beta("noise", 1.5, 6.0)  # share of examples believed random
    set_weights = pm.Dirichlet("set_weights", a=SET_CONCENTRATION)
    zero_split = pm.Beta("zero_split", 2, 2)
    shared = set_weights[END_0]
    rule_weights = pm.Deterministic("rule_weights", pt.concatenate([
        set_weights[:END_0], (zero_split * shared)[None], set_weights[END_0 + 1:CONTAINS_0],
        ((1 - zero_split) * shared)[None], set_weights[CONTAINS_0:],
    ]))
    ext_share = pm.Beta("ext_share", 1.0, 4.0)
    all_weights = pt.concatenate([(1 - ext_share) * rule_weights, pt.ones(N_EXT) * ext_share / N_EXT])

    sigma = pm.LogNormal("sigma", mu=np.log(20.0), sigma=1.0)
    log_erlang = pt.log(SIZES_F) - SIZES_F / sigma
    log_per_size = log_erlang - pt.logsumexp(log_erlang + np.log(N_PER_SIZE))
    log_prior = pt.concatenate([
        pt.log(lambda_0) + pt.log(all_weights),
        pt.log1p(-lambda_0) + log_per_size[INT_SIZE.astype("int64") - 1],
    ])  # (C,)

    # The one change: only range concepts draw their outputs with weight n**gamma;
    # rules draw their members uniformly (weight 1 each).
    gamma = pm.Normal("gamma", mu=0.0, sigma=1.0)
    w = pt.exp(gamma * LOG_N)  # (100,)
    int_mass = pt.dot(INT_MEMBERSHIP, w)  # (I,)
    w0 = pt.concatenate([w, pt.zeros(1)])
    subset_mass = pt.dot(w0[set_ex], SUBSETS.T).reshape((-1,))  # (sets * 16,)
    f32 = lambda x: pt.cast(x, "float32")  # the per-entry and per-concept work runs in float32
    is_int = f32(e_is_int)
    w_int = f32(w)[e_x]
    int_remaining = pt.maximum(f32(int_mass)[e_int] - f32(subset_mass)[e_prev], w_int)
    # Ranges: weighted share of the mass not yet shown; rules: 1 / members not yet shown.
    share = is_int * (w_int / int_remaining) + (1 - is_int) / pt.maximum(f32(e_rule_left), 1.0)
    out_term = f32(pt.log(noise / 100.0))
    noise32 = f32(noise)
    gain = pt.log((1 - noise32) * share + noise32 / 100.0) - out_term
    n_sets = n_examples.shape[0]
    log_lik = pt.inc_subtensor(pt.zeros(n_sets * N_CONC, dtype="float32")[e_flat], gain).reshape((n_sets, N_CONC))
    log_lik = log_lik + f32(n_examples)[:, None] * out_term
    log_joint = f32(log_prior)[None, :] + log_lik
    log_joint = log_joint - pt.max(log_joint, axis=1, keepdims=True)
    unnorm = pt.exp(log_joint)
    yes_table = pt.cast(pt.dot(unnorm, CONCEPT_MEMBERSHIP.astype("float32")) / unnorm.sum(axis=1, keepdims=True), "float64")  # (S, 100)
    p_yes = pm.Deterministic("p_yes", pt.clip(yes_table[row_set, row_target], 1e-9, 1 - 1e-9))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set`, `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    set_ex = np.full((len(unique), MAX_EX), 100, dtype="int64")
    flat, ints, is_int, left, xs, prevs = [], [], [], [], [], []
    for i, s in enumerate(unique):
        s = s[:MAX_EX]
        set_ex[i, :len(s)] = np.array(s) - 1
        for k, x in enumerate(s):
            cs = np.nonzero(CONCEPT_MEMBERSHIP[:, x - 1])[0]
            pv = np.full(len(cs), i * 2**MAX_EX, dtype="int64")
            n_prev = np.zeros(len(cs))
            for j, y in enumerate(s[:k]):
                inside = CONCEPT_MEMBERSHIP[cs, y - 1] > 0
                pv += inside.astype("int64") << j
                n_prev += inside
            rule = cs < N_RULES
            flat.append(i * N_CONC + cs)
            ints.append(np.where(rule, 0, cs - N_RULES))
            is_int.append((~rule).astype("float64"))
            left.append(np.where(rule, RULE_SIZE[np.minimum(cs, N_RULES - 1)] - n_prev, 1.0))
            xs.append(np.full(len(cs), x - 1))
            prevs.append(pv)
    return {
        "set_ex": set_ex,
        "n_examples": np.array([float(min(len(s), MAX_EX)) for s in unique]),
        "e_flat": np.concatenate(flat).astype("int64"),
        "e_int": np.concatenate(ints).astype("int64"),
        "e_is_int": np.concatenate(is_int).astype("float64"),
        "e_rule_left": np.concatenate(left).astype("float64"),
        "e_x": np.concatenate(xs).astype("int64"),
        "e_prev": np.concatenate(prevs).astype("int64"),
        "row_set": np.array([index[s] for s in sets], dtype="int64"),
        "row_target": np.array([int(r["target"]) - 1 for r in rows], dtype="int64"),
        "rating": np.array([int(r.get("rating", 0) or 0) for r in rows], dtype="int64"),
    }
