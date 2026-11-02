"""Pedagogical-sampling concept learning (rational pedagogy).

People do not assume the hidden program produced its examples at random from
its concept; they assume the examples were chosen to be helpful, as a teacher
would choose them. A teacher holding a concept picks each example in
proportion to how strongly that example, on its own, would lead a learner with
the same prior to that concept, and people judge each candidate concept by how
likely such a teacher would be to have picked the examples shown. The concept
space and prior are those of the best model (45 rules in families with their
own prior mass, bounded ranges with an Erlang prior on width, one-sided
thresholds as their own family), and "yes" is the posterior probability that
the inferred concept contains the target. So the most diagnostic examples
count most: a concept whose examples could equally well point to many other
concepts is discounted, even if it is small. The teacher is assumed to teach a
naive learner (equal prior mass per kind of concept, a default width prior),
so the teacher's choice of examples is fixed and computed once.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
ALPHA = 0.9
REFERENCE_SCALE = 20.0  # naive learner's typical range width


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


def family(name):
    if name in ("odd", "even"):
        return 0
    if name.startswith("multiples"):
        return 2
    if name.startswith("ends"):
        return 3
    if name.startswith("contains"):
        return 4
    if name.startswith("powers of 2,"):
        return 5  # gerrymandered exception rules
    return 1  # primes, squares, cubes, powers


_RULES = rule_concepts()
RULES = list(_RULES.values())
N_RULES = len(RULES)
RULE_FAMILY = np.array([family(n) for n in _RULES])
FAMILY_SIZE = np.bincount(RULE_FAMILY).astype("float64")
LO, HI = np.array([(m, n) for m in NUMBERS for n in range(m, 101)]).T
THRESHOLD = (LO == 1) | (HI == 100)
WIDTH = HI - LO + 1
WIDTHS = np.arange(1, 99)
N_PER_WIDTH = np.array([np.sum(~THRESHOLD & (WIDTH == w)) for w in WIDTHS], dtype="float64")
RANGE_WIDTH_INDEX = np.clip(WIDTH[~THRESHOLD] - 1, 0, len(WIDTHS) - 1)
# All concepts: rules, then bounded ranges, then thresholds.
MEMBERSHIP = np.concatenate(
    [
        np.array([np.isin(NUMBERS, c) for c in RULES]),
        (NUMBERS[None, :] >= LO[~THRESHOLD, None]) & (NUMBERS[None, :] <= HI[~THRESHOLD, None]),
        (NUMBERS[None, :] >= LO[THRESHOLD, None]) & (NUMBERS[None, :] <= HI[THRESHOLD, None]),
    ]
).astype("float64")
N_CONCEPTS, N_RANGES, N_THRESHOLD = len(MEMBERSHIP), int((~THRESHOLD).sum()), float(THRESHOLD.sum())
SIZE = MEMBERSHIP.sum(axis=1)

def _teacher_log_example():
    """log P(example x | concept h) for a teacher of a naive learner, mixed 10% random."""
    erlang = np.log(WIDTHS) - WIDTHS / REFERENCE_SCALE
    per_range = np.exp(erlang) / np.sum(np.exp(erlang) * N_PER_WIDTH)
    prior = np.concatenate(
        [
            (1.0 / 8.0 / FAMILY_SIZE)[RULE_FAMILY],
            per_range[RANGE_WIDTH_INDEX] / 8.0,
            np.full(int(N_THRESHOLD), 1.0 / 8.0 / N_THRESHOLD),
        ]
    )
    # A single example x leads the learner to h with P(h | x) = P(h) / |h| / Z(x);
    # the teacher picks x in h in proportion to it, i.e. to 1 / Z(x).
    inv_z = 1.0 / ((prior / SIZE) @ MEMBERSHIP)
    teach = MEMBERSHIP * inv_z[None, :] / (MEMBERSHIP @ inv_z)[:, None]
    return np.log(ALPHA * teach + (1 - ALPHA) / 100)


LOG_EXAMPLE = _teacher_log_example()  # (concepts, numbers)
RANGE_WIDTH = WIDTH[~THRESHOLD]

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    # Per set, log sum of P(D | range) over bounded ranges of each width and over all
    # thresholds, with each group's likelihood-weighted membership of every target.
    width_log_evidence = pm.Data("width_log_evidence", np.zeros((1, len(WIDTHS))))
    width_in_concept = pm.Data("width_in_concept", np.zeros((1, len(WIDTHS), 100)))
    threshold_log_evidence = pm.Data("threshold_log_evidence", np.zeros((1, 1)), shape=(None, 1))
    threshold_in_concept = pm.Data("threshold_in_concept", np.zeros((1, 100)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    family_weight = pm.Dirichlet("family_weight", a=np.ones(8))
    log_w = pt.log(family_weight)
    interval_scale = pm.LogNormal("interval_scale", mu=np.log(20.0), sigma=1.0)

    erlang = np.log(WIDTHS) - WIDTHS / interval_scale
    log_per_range = erlang - pt.logsumexp(erlang + np.log(N_PER_WIDTH))

    log_rules = (log_w[:6] - np.log(FAMILY_SIZE))[RULE_FAMILY][None, :] + rule_log_lik
    log_widths = log_w[6] + log_per_range[None, :] + width_log_evidence
    log_thresholds = log_w[7] - np.log(N_THRESHOLD) + threshold_log_evidence
    log_joint = pt.concatenate([log_rules, log_widths, log_thresholds], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = (
        posterior[:, :N_RULES] @ pt.as_tensor(MEMBERSHIP[:N_RULES])
        + (posterior[:, N_RULES:-1, None] * width_in_concept).sum(axis=1)
        + posterior[:, -1, None] * threshold_in_concept
    )  # (sets, targets)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def _group(log_lik, membership):
    """Log evidence of a group of equally likely concepts and its membership of each target."""
    top = log_lik.max(axis=1, keepdims=True)
    weight = np.exp(log_lik - top)
    total = weight.sum(axis=1)
    return np.log(total) + top[:, 0], (weight / total[:, None]) @ membership


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    count = np.zeros((len(unique), 100))
    for i, s in enumerate(unique):
        np.add.at(count[i], np.array(s) - 1, 1.0)
    log_lik = count @ LOG_EXAMPLE.T  # (sets, concepts)
    range_log_lik = log_lik[:, N_RULES : N_RULES + N_RANGES]
    range_membership = MEMBERSHIP[N_RULES : N_RULES + N_RANGES]
    width_log_evidence = np.zeros((len(unique), len(WIDTHS)))
    width_in_concept = np.zeros((len(unique), len(WIDTHS), 100))
    for i, w in enumerate(WIDTHS):
        sel = RANGE_WIDTH == w
        if sel.any():
            width_log_evidence[:, i], width_in_concept[:, i] = _group(range_log_lik[:, sel], range_membership[sel])
    threshold_log_evidence, threshold_in_concept = _group(
        log_lik[:, N_RULES + N_RANGES :], MEMBERSHIP[N_RULES + N_RANGES :]
    )
    return {
        "rule_log_lik": log_lik[:, :N_RULES],
        "width_log_evidence": width_log_evidence,
        "width_in_concept": width_in_concept,
        "threshold_log_evidence": threshold_log_evidence[:, None],
        "threshold_in_concept": threshold_in_concept,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
