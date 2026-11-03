"""The first example carries a different evidential weight from the later ones.

People infer the hidden program as the Bayesian concept learner of
`round_number_range_boundaries` does (45 number rules, bounded ranges with an
Erlang prior on width and a round-number bias on range boundaries, one-sided
thresholds as their own family, a prior mass per concept family, examples drawn
without replacement, 10% of them taken as random draws). The one change is
order-dependent: the first example shown, met before people have any guess
about the program, revises their belief with its own evidential weight -- its
likelihood is raised to exp(first_example_log_weight) -- while every later
example counts at the normative weight of one. A weight below one means the
opening example is only a loose starting point that later examples refine
(the "suspicious coincidence" it provides is discounted); above one means a
first impression that dominates; zero log-weight recovers the ideal learner.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
ALPHA = 0.9


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
        return 5
    return 1


_RULES = rule_concepts()
RULES = list(_RULES.values())
N_RULES = len(RULES)
RULE_FAMILY = np.array([family(n) for n in _RULES])
FAMILY_SIZE = np.bincount(RULE_FAMILY).astype("float64")
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
LO, HI = np.array([(m, n) for m in NUMBERS for n in range(m, 101)]).T
THRESHOLD = (LO == 1) | (HI == 100)
INTERVAL_MEMBERSHIP = ((NUMBERS[None, :] >= LO[:, None]) & (NUMBERS[None, :] <= HI[:, None])).astype("float64")
WIDTH = HI - LO + 1
ROUND_EDGES = ((LO % 10) == 0).astype(int) + ((HI % 10) == 0).astype(int)
# Ranges are grouped so that the prior is uniform within a group: bounded ranges by
# (width, number of round edges), thresholds by size. Within a group, every member
# containing the first example gives it the same likelihood, so each group is split,
# per set, into the members that do and do not contain the first example.
GROUPS = [("b", w, r) for w in range(1, 99) for r in range(3)
          if np.any(~THRESHOLD & (WIDTH == w) & (ROUND_EDGES == r))]
GROUPS += [("t", w, 0) for w in range(1, 101)]
GROUP_SEL = [(~THRESHOLD if k == "b" else THRESHOLD) & (WIDTH == w) & ((ROUND_EDGES == r) | (k == "t"))
             for k, w, r in GROUPS]
G = len(GROUPS)
IS_BOUNDED = np.array([k == "b" for k, _, _ in GROUPS])
G_WIDTH = np.array([w for _, w, _ in GROUPS], dtype="float64")
G_ROUND = np.array([r for _, _, r in GROUPS], dtype="float64")
G_COUNT = np.array([sel.sum() for sel in GROUP_SEL], dtype="float64")
N_THRESHOLD = float(THRESHOLD.sum())
B_IDX = np.where(IS_BOUNDED)[0]
T_IDX = np.where(~IS_BOUNDED)[0]
# Log-likelihood of the first example under a concept of a given size that contains it,
# and under one that does not.
FIRST_OUT = np.log((1 - ALPHA) / 100)
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
NEG = -700.0

with pm.Model() as model:
    rule_first = pm.Data("rule_first", np.zeros((1, N_RULES)))  # first-example log-lik
    rule_rest = pm.Data("rule_rest", np.zeros((1, N_RULES)))  # later examples' log-lik
    rule_member = pm.Data("rule_member", np.zeros((1, N_RULES, 1)))
    # Per set and group: log sum of later-example likelihoods over the members that do
    # (in) / do not (out) contain the first example, and their posterior membership of
    # each asked target.
    ev_in = pm.Data("ev_in", np.zeros((1, G)))
    ev_out = pm.Data("ev_out", np.zeros((1, G)))
    member_in = pm.Data("member_in", np.zeros((1, G, 1)))
    member_out = pm.Data("member_out", np.zeros((1, G, 1)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    family_weight = pm.Dirichlet("family_weight", a=np.ones(8))
    log_w = pt.log(family_weight)
    interval_scale = pm.LogNormal("interval_scale", mu=np.log(20.0), sigma=1.0)
    round_bonus = pm.Normal("round_bonus", 0.0, 1.0)
    # Log evidential weight of the first example shown (later examples weigh 1).
    first_example_log_weight = pm.Normal("first_example_log_weight", 0.0, 0.3)
    wf = pt.exp(first_example_log_weight)

    erlang = np.log(G_WIDTH[B_IDX]) - G_WIDTH[B_IDX] / interval_scale + round_bonus * G_ROUND[B_IDX]
    log_per_bounded = log_w[6] + erlang - pt.logsumexp(erlang + np.log(G_COUNT[B_IDX]))
    log_per_concept = pt.zeros(G)
    log_per_concept = pt.set_subtensor(log_per_concept[B_IDX], log_per_bounded)
    log_per_concept = pt.set_subtensor(log_per_concept[T_IDX], log_w[7] - np.log(N_THRESHOLD))

    first_in = np.log(ALPHA / G_WIDTH + (1 - ALPHA) / 100)
    log_in = log_per_concept[None, :] + wf * first_in[None, :] + ev_in
    log_out = log_per_concept[None, :] + wf * FIRST_OUT + ev_out
    log_rules = (log_w[:6] - np.log(FAMILY_SIZE))[RULE_FAMILY][None, :] + wf * rule_first + rule_rest
    log_joint = pt.concatenate([log_rules, log_in, log_out], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    members = pt.concatenate([rule_member, member_in, member_out], axis=1)
    in_concept = (posterior[:, :, None] * members).sum(axis=1)  # (sets, asked targets)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def _step_log_lik(examples, membership):
    """Per-example log P(example | concept, earlier examples), sampling without replacement."""
    size = membership.sum(axis=1)
    shown = np.zeros(len(membership))
    out = []
    for x in examples:
        inside = membership[:, x - 1]
        left = np.maximum(size - shown, 1.0)
        out.append(np.log(ALPHA * inside / left + (1 - ALPHA) / 100))
        shown += inside
    return np.array(out)


def _group(log_lik, membership):
    if len(log_lik) == 0:
        return NEG, np.zeros(membership.shape[1])
    top = log_lik.max()
    weight = np.exp(log_lik - top)
    return np.log(weight.sum()) + top, (weight / weight.sum()) @ membership


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    asked = [sorted({int(r["target"]) for s, r in zip(sets, rows) if s == u}) for u in unique]
    slot = [{t: j for j, t in enumerate(a)} for a in asked]
    MAX_T = max(len(a) for a in asked)
    n = len(unique)
    out = {k: np.zeros(shape) for k, shape in [
        ("rule_first", (n, N_RULES)), ("rule_rest", (n, N_RULES)), ("rule_member", (n, N_RULES, MAX_T)),
        ("ev_in", (n, G)), ("ev_out", (n, G)), ("member_in", (n, G, MAX_T)), ("member_out", (n, G, MAX_T))]}
    for i, u in enumerate(unique):
        cols = np.array(asked[i]) - 1
        rl = _step_log_lik(u, RULE_MEMBERSHIP)
        out["rule_first"][i] = rl[0]
        out["rule_rest"][i] = rl[1:].sum(axis=0)
        out["rule_member"][i, :, : len(cols)] = RULE_MEMBERSHIP[:, cols]
        il = _step_log_lik(u, INTERVAL_MEMBERSHIP)[1:].sum(axis=0)
        contains_first = INTERVAL_MEMBERSHIP[:, u[0] - 1] > 0
        for g, sel in enumerate(GROUP_SEL):
            for key, part in (("in", sel & contains_first), ("out", sel & ~contains_first)):
                ev, mem = _group(il[part], INTERVAL_MEMBERSHIP[part][:, cols])
                out["ev_" + key][i, g] = ev
                out["member_" + key][i, g, : len(cols)] = mem
    out["row_cell"] = np.array([index[s] * MAX_T + slot[index[s]][int(r["target"])] for s, r in zip(sets, rows)], dtype="int64")
    out["rating"] = np.array([int(r["rating"]) for r in rows], dtype="int64")
    return out
