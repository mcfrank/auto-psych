"""Sequential win-stay, lose-shift revision of a single concept guess.

Hypothesis: people hold one working guess about the concept and revise it
example by example in the order shown. After the first example they pick a
concept (a rule or a moderate-width range) in proportion to its plausibility;
when a later example fits the current guess they keep it unexamined, and only
when an example contradicts it do they (at a fitted rate) drop it and re-pick a
concept in light of all examples seen so far. Early examples therefore lock in
broad concepts that later consistent examples never tighten, and order matters.

Concept space and plausibility are those of `final_digit_salient_rules` (45
rules with final-digit rules boosted by kappa, Erlang-width intervals sampled
on a compressed number line, tempered size principle beta, exception rate eps).
The distribution over the guess after example t is
D_t = D_{t-1} * stay_t + (1 - sum(D_{t-1} * stay_t)) * P_t, where P_t is the
posterior given examples 1..t and stay_t(h) = 1 if example t is in h, else
1 - shift (shift = probability of abandoning a contradicted guess).
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_N = 4  # most examples in a set


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


RULE_NAMES = list(rule_concepts().keys())
RULES = list(rule_concepts().values())
RULE_SALIENT = np.array([n in ("odd", "even") or n.startswith("ends in") for n in RULE_NAMES], dtype="float64")
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
INT_A = np.array([a for a in NUMBERS for b in NUMBERS if b >= a])
INT_B = np.array([b for a in NUMBERS for b in NUMBERS if b >= a])
INT_MEMBERSHIP = ((INT_A[:, None] <= NUMBERS[None, :]) & (INT_B[:, None] >= NUMBERS[None, :])).astype("float64")
INT_WIDTH_IDX = INT_B - INT_A  # width - 1
LOG_NUMBERS = np.log(NUMBERS.astype("float64"))
SIZES = np.arange(1, 101, dtype="float64")
LOG_SIZE_COUNT = np.log(101 - SIZES)
# (concepts, numbers): membership of every number in every concept, rules first.
MEMBER = np.concatenate([RULE_MEMBERSHIP, INT_MEMBERSHIP])
N_CONCEPTS = MEMBER.shape[0]


with pm.Model() as model:
    # Sets are sorted by size, largest first. For the prefix of the first t+1 examples, one row
    # per set with more than t examples (sets, concepts), float32 for speed: examples so far
    # inside each concept, the summed log of those inside an interval (its sampling term), and
    # (t >= 1) whether example t falls outside the concept.
    pre_k = [pm.Data(f"pre_k_{t}", np.zeros((1, N_CONCEPTS), dtype="float32")) for t in range(MAX_N)]
    pre_log_x = [pm.Data(f"pre_log_x_{t}", np.zeros((1, N_CONCEPTS), dtype="float32")) for t in range(MAX_N)]
    new_out = [None] + [pm.Data(f"new_out_{t}", np.zeros((1, N_CONCEPTS), dtype="float32")) for t in range(1, MAX_N)]
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_target = pm.Data("cell_target", np.zeros(1, dtype="int64"))  # target - 1
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.5)  # sharpness of the size principle
    eps = pm.Beta("eps", 2, 5)  # believed rate of off-concept examples
    gamma = pm.HalfNormal("gamma", sigma=1.0)  # compression of the mental number line
    sigma = pm.LogNormal("sigma", mu=np.log(20.0), sigma=1.0)  # typical interval width
    kappa = pm.Normal("kappa", mu=0.0, sigma=1.5)  # log prior boost of final-digit rules
    shift = pm.Beta("shift", 2, 2)  # probability of abandoning a guess an example contradicts

    log_noise = pt.log(eps) - np.log(100)

    # Prior over concepts.
    log_rule_prior = pt.log(lambda_0) + kappa * RULE_SALIENT - pm.math.logsumexp(kappa * RULE_SALIENT)
    log_width = pt.log(SIZES) - SIZES / sigma
    log_z = pm.math.logsumexp(LOG_SIZE_COUNT + log_width)
    log_int_prior = (pt.log1p(-lambda_0) + log_width - log_z)[INT_WIDTH_IDX]
    log_prior = pt.concatenate([log_rule_prior, log_int_prior])  # (concepts,)

    # Log probability of an in-concept example, apart from the -gamma*log x term of intervals.
    rule_in = pt.log((1 - eps) / RULE_SIZE + eps / 100)
    cum_w = pt.concatenate([pt.zeros(1), pt.cumsum(pt.exp(-gamma * LOG_NUMBERS))])
    int_in = pt.log1p(-eps) - pt.log(cum_w[INT_B] - cum_w[INT_A - 1])
    log_in = pt.concatenate([rule_in, int_in])  # (concepts,)

    # Posterior after each prefix of the examples. The examples' shared noise term
    # n * log_noise is the same for every concept and cancels in the softmax.
    f32 = "float32"
    gain = pt.cast(beta * (log_in - log_noise), f32)[None, :]
    prior32 = pt.cast(log_prior, f32)[None, :]
    compress = pt.cast(beta * gamma, f32)

    def posterior(t):
        return pt.special.softmax(prior32 + pre_k[t] * gain - compress * pre_log_x[t], axis=1)

    # Win-stay, lose-shift over the examples in order (only sets with an example t take step t).
    shift32 = pt.cast(shift, f32)
    dist = posterior(0)
    for t in range(1, MAX_N):
        rows = new_out[t].shape[0]
        kept = dist[:rows] * (1 - shift32 * new_out[t])
        stepped = kept + (1 - kept.sum(axis=1))[:, None] * posterior(t)
        dist = pt.concatenate([stepped, dist[rows:]], axis=0)

    in_concept = pt.cast(pt.dot(dist, MEMBER.astype(f32)), "float64")[cell_set, cell_target]  # (cells,)
    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = sorted(dict.fromkeys(sets), key=len, reverse=True)
    index = {s: i for i, s in enumerate(unique)}
    trial_cells = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(trial_cells))
    cell_index = {c: i for i, c in enumerate(cells)}
    n_rows = [sum(len(s) > t for s in unique) for t in range(MAX_N)]
    pre_k = [np.zeros((n, N_CONCEPTS), dtype="float32") for n in n_rows]
    pre_log_x = [np.zeros((n, N_CONCEPTS), dtype="float32") for n in n_rows]
    new_out = [np.zeros((n, N_CONCEPTS), dtype="float32") for n in n_rows]
    is_int = np.concatenate([np.zeros(N_RULES), np.ones(len(INT_A))])
    for i, s in enumerate(unique):
        k = np.zeros(N_CONCEPTS)
        lx = np.zeros(N_CONCEPTS)
        for t, x in enumerate(s):
            inside = MEMBER[:, x - 1]
            k = k + inside
            lx = lx + is_int * inside * np.log(x)
            pre_k[t][i], pre_log_x[t][i], new_out[t][i] = k, lx, 1.0 - inside
    data = {f"pre_k_{t}": pre_k[t] for t in range(MAX_N)}
    data.update({f"pre_log_x_{t}": pre_log_x[t] for t in range(MAX_N)})
    data.update({f"new_out_{t}": new_out[t] for t in range(1, MAX_N)})
    return {
        **data,
        "cell_set": np.array([i for i, _ in cells], dtype="int64"),
        "cell_target": np.array([t - 1 for _, t in cells], dtype="int64"),
        "row_cell": np.array([cell_index[c] for c in trial_cells], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
