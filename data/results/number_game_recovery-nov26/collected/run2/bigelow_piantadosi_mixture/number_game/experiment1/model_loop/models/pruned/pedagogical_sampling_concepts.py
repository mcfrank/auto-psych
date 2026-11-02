"""Pedagogical sampling: people treat the examples as chosen by a helpful
teacher, not drawn at random from the concept. The teacher picks, among the
concept's members, numbers that would most lead a learner to that concept --
P_teach(x | h) is proportional to P(h | x)^teach over the members x of h, where
P(h | x) is a size-principle learner's belief after seeing x alone -- and the
participant infers the concept by inverting this teacher. Hypotheses: the 45
arithmetic and digit rules (equal rule prior lambda_0) and every interval of
1..100, 10% of examples random, plus a guess lapse (rates of guessed yes and
guessed no). teach = 0 recovers the size principle; teach > 0 makes examples
that few concepts contain more diagnostic.

The teacher's imagined learner splits its prior evenly between rules and
intervals. Because single-example likelihoods are constant within a concept,
P(h | x)^teach over members reduces to u_x = Z_x^-teach, Z_x the imagined
learner's marginal probability of x. Each set's (and each set-target cell's)
summed rule and interval likelihoods then depend on teach alone; they are
tabulated per stimulus in prepare_observed as Chebyshev series in teach
(log scale), and lambda_0 enters exactly.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

ALPHA = 0.9
FLOOR = (1 - ALPHA) / 100
NUMBERS = np.arange(1, 101)
TEACH_MAX = 3.0  # series domain [0, TEACH_MAX]; the prior puts < 1e-6 beyond it
N_COEF = 16


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
    return list(rules.values())


RULES = rule_concepts()
N_RULES = len(RULES)
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
N_INTERVALS = len(INTERVALS)
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
BASE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + FLOOR)
# Imagined learner's log P(x), prior split evenly between rules and intervals.
LOG_Z = np.logaddexp(
    np.log(0.5 / N_RULES) + logsumexp(BASE_LOG_LIK[:N_RULES], axis=0),
    np.log(0.5 / N_INTERVALS) + logsumexp(BASE_LOG_LIK[N_RULES:], axis=0),
)
NODES = 0.5 * TEACH_MAX * (1 + np.cos(np.pi * (np.arange(N_COEF) + 0.5) / N_COEF))  # Chebyshev nodes


def class_log_sums(counts, teach):
    """For each set (rows of `counts`, 100 example counts), log of the summed
    teacher likelihood over rules and over intervals, in total and restricted
    to hypotheses containing each target: arrays (sets,), (sets,), (sets, 100) x2."""
    log_u = -teach * LOG_Z
    u = np.exp(log_u - log_u.max())
    p = MEMBERSHIP * (ALPHA * u[None, :] / (MEMBERSHIP @ u)[:, None]) + FLOOR
    log_lik = counts @ np.log(p).T  # (sets, H)
    out = []
    for part in (slice(0, N_RULES), slice(N_RULES, None)):
        ll = log_lik[:, part]
        top = ll.max(axis=1, keepdims=True)
        w = np.exp(ll - top)
        out.append(np.log(w.sum(axis=1)) + top[:, 0])
        out.append(np.log(w @ MEMBERSHIP[part]) + top)
    return out


def chebyshev_basis(y, n):
    """T_0..T_{n-1} at y in [-1, 1] (numpy or pytensor)."""
    terms = [y * 0 + 1, y]
    for _ in range(n - 2):
        terms.append(2 * y * terms[-1] - terms[-2])
    return terms[:n]


NODE_BASIS = np.stack(chebyshev_basis(2 * NODES / TEACH_MAX - 1, N_COEF), axis=1)  # (nodes, coef)

with pm.Model() as model:
    # Chebyshev coefficients (in teach) of the log summed likelihoods.
    set_rule_coef = pm.Data("set_rule_coef", np.zeros((1, N_COEF)))
    set_interval_coef = pm.Data("set_interval_coef", np.zeros((1, N_COEF)))
    cell_rule_coef = pm.Data("cell_rule_coef", np.zeros((1, N_COEF)))
    cell_interval_coef = pm.Data("cell_interval_coef", np.zeros((1, N_COEF)))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    teach = pm.HalfNormal("teach", sigma=0.5)  # strength of the helpful-teacher assumption
    log_prior_rule = pt.log(lambda_0) - np.log(N_RULES)
    log_prior_interval = pt.log1p(-lambda_0) - np.log(N_INTERVALS)

    basis = pt.stack(chebyshev_basis(2 * pt.minimum(teach, TEACH_MAX) / TEACH_MAX - 1, N_COEF))
    log_evidence = pt.logaddexp(
        log_prior_rule + pt.dot(set_rule_coef, basis), log_prior_interval + pt.dot(set_interval_coef, basis)
    )
    log_in_target = pt.logaddexp(
        log_prior_rule + pt.dot(cell_rule_coef, basis), log_prior_interval + pt.dot(cell_interval_coef, basis)
    )
    in_concept = pt.exp(log_in_target - log_evidence[cell_set])

    p_concept = pt.clip(in_concept[row_cell], 0.0, 1.0)
    # Guess lapse, as the rates of guessed "yes" and guessed "no" trials.
    guess_yes = pm.Beta("guess_yes", 1.5, 20)
    guess_no = pm.Beta("guess_no", 1.5, 20)
    p_yes = pm.Deterministic("p_yes", pt.clip((1 - guess_yes - guess_no) * p_concept + guess_yes, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    set_index = {s: i for i, s in enumerate(unique)}
    cells = list(dict.fromkeys((s, int(r["target"])) for s, r in zip(sets, rows)))
    cell_index = {c: i for i, c in enumerate(cells)}
    cell_s = np.array([set_index[s] for s, _ in cells], dtype="int64")
    cell_t = np.array([t - 1 for _, t in cells], dtype="int64")
    counts = np.zeros((len(unique), 100))
    for i, s in enumerate(unique):
        for x in s:
            counts[i, x - 1] += 1
    at_nodes = [[], [], [], []]
    for teach in NODES:
        rule, rule_in, interval, interval_in = class_log_sums(counts, teach)
        for store, value in zip(at_nodes, (rule, rule_in[cell_s, cell_t], interval, interval_in[cell_s, cell_t])):
            store.append(value)
    fit = lambda values: np.linalg.solve(NODE_BASIS, np.stack(values)).T  # noqa: E731
    return {
        "set_rule_coef": fit(at_nodes[0]),
        "cell_rule_coef": fit(at_nodes[1]),
        "set_interval_coef": fit(at_nodes[2]),
        "cell_interval_coef": fit(at_nodes[3]),
        "cell_set": cell_s,
        "row_cell": np.array([cell_index[(s, int(r["target"]))] for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
