"""Win-stay, lose-shift concept revision: people hold one running guess about
the concept and revise it example by example, in the order shown. After the
first example they adopt one concept sampled from the posterior (42 rules with
a kind-level prior and rule share p_mix, every interval of 1-100 with an
Erlang width prior and a decade-bounded preference, size principle with 10%
uniform examples). A later example the current guess contains keeps the guess;
an example it does not contain triggers a fresh draw from the posterior given
all examples seen so far. "Yes" = probability the guess held after the last
example contains the target. Order of examples matters.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

EPSILON = 0.1
NUMBERS = np.arange(1, 101)
MAX_LEN = 4


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
    kinds = [0] * 9 + [1] * 10 + [2] * 9 + [3] * 9 + [4] * 5
    bounds = [(m, n) for m in range(1, 101) for n in range(m, 101)]
    return rules, kinds, bounds


RULES, RULE_KINDS, BOUNDS = concept_sets()
N_RULES = len(RULES)
N_KINDS = 5
KIND_MATRIX = np.eye(N_KINDS)[RULE_KINDS]
KIND_COUNTS = KIND_MATRIX.sum(axis=0)
INTERVALS = [list(range(m, n + 1)) for m, n in BOUNDS]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
N_HYP = MEMBERSHIP.shape[0]
EXAMPLE_LOG_LIK = np.log(MEMBERSHIP * (1 - EPSILON) / MEMBERSHIP.sum(axis=1, keepdims=True) + EPSILON / 100)

INTERVAL_WIDTH = np.array([n - m + 1 for m, n in BOUNDS])
DECADE_BOUNDED = np.array([(m - 1) % 10 == 0 and n % 10 == 0 for m, n in BOUNDS], dtype="int64")
# Interval priors depend only on (width, decade-bounded): pool intervals into groups.
GROUP_KEYS, GROUP_OF_INTERVAL, GROUP_SIZES = np.unique(
    np.stack([INTERVAL_WIDTH, DECADE_BOUNDED], axis=1), axis=0, return_inverse=True, return_counts=True
)
GROUP_OF_INTERVAL = GROUP_OF_INTERVAL.ravel()
GROUP_WIDTHS = GROUP_KEYS[:, 0].astype("float64")
GROUP_ALIGNED = GROUP_KEYS[:, 1].astype("float64")
N_GROUPS = len(GROUP_KEYS)
GROUP_INDICATOR = np.eye(N_GROUPS)[GROUP_OF_INTERVAL].T  # (G, n_intervals)
N_COMPACT = N_RULES + N_GROUPS

# The win-stay/lose-shift recursion, with g_1 = pi_1 and
#   g_k(h) = g_{k-1}(h) c_k(h) + M_k pi_k(h),  M_k = sum_h g_{k-1}(h) (1 - c_k(h)),
# expands into sums of the form sum_h pi_j(h) f(h) for fixed 0/1 functions f.
# Each is computed as (posterior over rules and interval groups after prefix j)
# dot (precomputed within-group likelihood-weighted fraction of f).

with pm.Model() as model:
    prefix_rule_log_lik = pm.Data("prefix_rule_log_lik", np.zeros((1, N_RULES)))
    prefix_group_log_evidence = pm.Data("prefix_group_log_evidence", np.zeros((1, N_GROUPS)))
    set_prefix = pm.Data("set_prefix", np.zeros((1, MAX_LEN), dtype="int64"))
    set_valid = pm.Data("set_valid", np.zeros((1, MAX_LEN)))
    # shift_frac[s, j, k]: share of pi_j kept through examples j+1..k-1 and rejected by k.
    shift_frac = pm.Data("shift_frac", np.zeros((1, MAX_LEN, MAX_LEN, N_COMPACT)))
    # yes_frac[c, j]: share of pi_j kept through all later examples and containing the target.
    yes_frac = pm.Data("yes_frac", np.zeros((1, MAX_LEN, N_COMPACT)))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 2, 2)
    kind_weights = pm.Dirichlet("kind_weights", a=np.full(N_KINDS, 2.0))
    rule_weights = pt.dot(KIND_MATRIX, kind_weights / KIND_COUNTS)
    width_scale = pm.LogNormal("width_scale", mu=np.log(15.0), sigma=1.0)
    decade_boost = pm.Normal("decade_boost", mu=1.0, sigma=1.5)

    log_interval = np.log(GROUP_WIDTHS) - GROUP_WIDTHS / width_scale + decade_boost * GROUP_ALIGNED
    log_interval_prior = log_interval - pt.logsumexp(log_interval + np.log(GROUP_SIZES))

    log_rules = pt.log(p_mix) + pt.log(rule_weights)[None, :] + prefix_rule_log_lik
    log_groups = pt.log1p(-p_mix) + log_interval_prior[None, :] + prefix_group_log_evidence
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))

    q = posterior[set_prefix]  # (n_sets, MAX_LEN, N_COMPACT)
    weights = [pt.ones(q.shape[0])]  # A_1 = 1, A_k = M_k (mass shifted at example k)
    for k in range(1, MAX_LEN):
        shifted = sum(weights[j] * (q[:, j] * shift_frac[:, j, k]).sum(axis=1) for j in range(k))
        weights.append(set_valid[:, k] * shifted)
    weights = pt.stack(weights, axis=1)  # (n_sets, MAX_LEN)

    cell_yes = (weights[cell_set] * (q[cell_set] * yes_frac).sum(axis=2)).sum(axis=1)
    p_yes = pm.Deterministic("p_yes", pt.clip(cell_yes[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def _fraction(weights, f):
    """Rules: f itself; interval groups: likelihood-weighted share of f in each group."""
    return np.concatenate([f[:N_RULES], GROUP_INDICATOR @ (weights * f[N_RULES:])])


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    distinct = list(dict.fromkeys(sets))
    set_index = {s: i for i, s in enumerate(distinct)}
    cells = list(dict.fromkeys(zip(sets, targets)))
    cell_index = {c: i for i, c in enumerate(cells)}

    prefixes = {}
    set_prefix = np.zeros((len(distinct), MAX_LEN), dtype="int64")
    set_valid = np.zeros((len(distinct), MAX_LEN))
    for i, s in enumerate(distinct):
        for k in range(MAX_LEN):
            key = tuple(sorted(s[: min(k, len(s) - 1) + 1]))
            prefixes.setdefault(key, len(prefixes))
            set_prefix[i, k] = prefixes[key]
            set_valid[i, k] = float(k < len(s))
    prefix_ll = np.stack([EXAMPLE_LOG_LIK[:, np.array(p) - 1].sum(axis=1) for p in prefixes])
    interval_ll = prefix_ll[:, N_RULES:]
    group_log_ev = np.stack([logsumexp(interval_ll[:, GROUP_OF_INTERVAL == g], axis=1) for g in range(N_GROUPS)], axis=1)
    within = np.exp(interval_ll - group_log_ev[:, GROUP_OF_INTERVAL])  # normalized within group

    shift_frac = np.zeros((len(distinct), MAX_LEN, MAX_LEN, N_COMPACT))
    set_survive = {}
    for i, s in enumerate(distinct):
        n = len(s)
        for j in range(n):
            w = within[set_prefix[i, j]]
            keep = np.ones(N_HYP)
            for k in range(j + 1, n):
                c = MEMBERSHIP[:, s[k] - 1]
                shift_frac[i, j, k] = _fraction(w, keep * (1 - c))
                keep = keep * c
            set_survive[(i, j)] = (w, keep)

    yes_frac = np.zeros((len(cells), MAX_LEN, N_COMPACT))
    for c, (s, t) in enumerate(cells):
        i = set_index[s]
        for j in range(len(s)):
            w, keep = set_survive[(i, j)]
            yes_frac[c, j] = _fraction(w, keep * MEMBERSHIP[:, t - 1])

    return {
        "prefix_rule_log_lik": prefix_ll[:, :N_RULES],
        "prefix_group_log_evidence": group_log_ev,
        "set_prefix": set_prefix,
        "set_valid": set_valid,
        "shift_frac": shift_frac,
        "yes_frac": yes_frac,
        "cell_set": np.array([set_index[s] for s, _ in cells], dtype="int64"),
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
