"""Refinement of fuzzy_trace_example_memory: win-stay, lose-shift concept
tracking over fuzzy memory traces of the examples.

People do Bayesian concept learning over 45 rules and every interval of 1..100
(size principle, 10% noisy examples, rules sharing prior mass lambda_0 equally,
an Erlang prior on interval size), and each example is remembered verbatim only
with probability r, otherwise only as its gist (its magnitude to within +-5), so
the trace likelihood is r * p(e | h) + (1 - r) * mean_{|x - e| <= 5} p(x | h).
The one change (from first_guess_win_stay_concepts): people hold a single
working guess, drawn after the first example from the trace posterior; as each
further example arrives the guess is kept with probability `stay` if the
remembered example fits it (verbatim: e in h; gist: the remembered value, drawn
from the window, lies in h), otherwise it is redrawn from the trace posterior
given all examples so far. "Yes" is the probability the final guess contains
the target.

For speed, intervals are grouped per example set by (size, containment of and
overlap with each example's window), which fixes every likelihood and fit, and
the (set, hypothesis) pairs are laid out flat with per-set segment sums.
"""

import numpy as np
import pymc as pm
import pytensor.sparse as ps
import pytensor.tensor as pt

ALPHA = 0.9
NUMBERS = np.arange(1, 101)
WINDOW = 5
MAX_N = 4


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


RULES = list(rule_concepts().values())
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
LO = np.array([m for m in NUMBERS for n in range(m, 101)])
HI = np.array([n for m in NUMBERS for n in range(m, 101)])
INTERVAL_SIZE = (HI - LO + 1).astype("float64")
INTERVAL_MEMBERSHIP = ((NUMBERS[None, :] >= LO[:, None]) & (NUMBERS[None, :] <= HI[:, None])).astype("float64")
MEMBERSHIP = np.concatenate([RULE_MEMBERSHIP, INTERVAL_MEMBERSHIP], axis=0)  # (hyp, 100)
N_HYP = MEMBERSHIP.shape[0]
EX_LIK = ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100
BLUR = np.array([(np.abs(NUMBERS - e) <= WINDOW) / (np.abs(NUMBERS - e) <= WINDOW).sum() for e in NUMBERS])
GIST_LIK = EX_LIK @ BLUR.T  # (hyp, 100): mean p(x | h) over the window of each value
GIST_FIT = MEMBERSHIP @ BLUR.T  # (hyp, 100): share of the window of each value lying in h

with pm.Model() as model:
    exact_lik = pm.Data("exact_lik", np.ones((MAX_N, 1)))  # p(e_k | h) per (set, hyp) instance; 1 if missing
    gist_lik = pm.Data("gist_lik", np.ones((MAX_N, 1)))
    exact_fit = pm.Data("exact_fit", np.ones((MAX_N, 1)))  # e_k in h
    gist_fit = pm.Data("gist_fit", np.ones((MAX_N, 1)))  # share of e_k's window in h
    step_mask = pm.Data("step_mask", np.zeros((MAX_N, 1)))  # example k exists for the instance's set
    inst_set = pm.Data("inst_set", np.zeros(1, dtype="int64"))
    inst_is_rule = pm.Data("inst_is_rule", np.ones(1))
    inst_size = pm.Data("inst_size", np.ones(1))  # interval size (1 for rules)
    inst_log_count = pm.Data("inst_log_count", np.zeros(1))  # log number of intervals in the group
    set_ptr = pm.Data("set_ptr", np.array([0, 1], dtype="int64"))  # instance offsets of each set
    cell_dummy = pm.Data("cell_dummy", np.zeros(1))  # one entry per asked (set, target) cell
    nz_inst = pm.Data("nz_inst", np.zeros(1, dtype="int64"))  # sparse (cell, instance) target membership
    nz_cell = pm.Data("nz_cell", np.zeros(1, dtype="int64"))
    nz_share = pm.Data("nz_share", np.zeros(1))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    log_sigma = pm.Normal("log_sigma", mu=np.log(10.0), sigma=1.0)
    sigma = pt.exp(log_sigma)
    # Probability that an example is remembered verbatim rather than as gist.
    r = pm.Beta("r", 2, 2)
    # Probability of keeping a guess that the remembered new example fits.
    stay = pm.Beta("stay", 2, 2)

    size = pt.as_tensor(INTERVAL_SIZE)
    log_norm = pt.logsumexp(pt.log(size) - size / sigma)
    log_prior = (inst_is_rule * (pt.log(lambda_0) - np.log(N_RULES))
                 + (1 - inst_is_rule) * (pt.log1p(-lambda_0) + inst_log_count + pt.log(inst_size)
                                         - inst_size / sigma - log_norm))
    prefix_ll = pt.cumsum(pt.log(r * exact_lik + (1 - r) * gist_lik), axis=0)  # (steps, inst)
    joint = pt.exp(log_prior[None, :] + prefix_ll)  # all terms <= 0, rules keep it well above underflow
    n_inst = inst_set.shape[0]
    # (sets, instances) 0/1 matrix summing each set's contiguous block of instances.
    set_sum = ps.CSR(pt.ones(n_inst), pt.arange(n_inst).astype("int32"), set_ptr.astype("int32"),
                     pt.stack([set_ptr.shape[0] - 1, n_inst]).astype("int32"))

    def seg_sum(x):  # x: (k, instances) -> (k, sets)
        return ps.structured_dot(set_sum, x.T).T

    set_spread = ps.transpose(set_sum)

    def spread(t):  # t: (k, sets) -> (k, instances), each set's value copied to its instances
        return ps.structured_dot(set_spread, t.T).T

    def per_set_normalise(x):
        return x / spread(seg_sum(x))

    post = per_set_normalise(joint)
    fit = r * exact_fit + (1 - r) * gist_fit

    q = post[0]
    for k in range(1, MAX_N):
        kept = stay * q * fit[k]
        kept_tot = seg_sum(kept[None, :])[0]
        updated = kept + (1 - spread(kept_tot[None, :])[0]) * post[k]
        m = step_mask[k]
        q = m * updated + (1 - m) * q

    in_cell = pt.inc_subtensor(pt.zeros(cell_dummy.shape[0])[nz_cell], q[nz_inst] * nz_share)
    p_yes = pm.Deterministic("p_yes", pt.clip(in_cell[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    blocks, offsets, members = [], [], []
    offset = 0
    for i, s in enumerate(unique):
        cols = [INTERVAL_SIZE.astype(int)]
        for e in s:
            cols.append(((LO <= e) & (HI >= e)).astype(int))
            cols.append((INTERVAL_MEMBERSHIP @ BLUR[e - 1] * 1000).round().astype(int))
        keys, group = np.unique(np.stack(cols, axis=1), axis=0, return_inverse=True)
        group = group.reshape(-1)
        n_g = len(keys)
        first = np.zeros(n_g, dtype="int64")
        first[group[::-1]] = np.arange(len(group))[::-1]  # a representative interval per group
        count = np.bincount(group, minlength=n_g).astype("float64")
        g_member = np.zeros((n_g, 100))
        np.add.at(g_member, group, INTERVAL_MEMBERSHIP)
        g_member /= count[:, None]
        hyp = np.concatenate([np.arange(N_RULES), N_RULES + first])  # representative hypothesis rows
        n_inst = len(hyp)
        b = {k: np.ones((MAX_N, n_inst)) for k in ["exact_lik", "gist_lik", "exact_fit", "gist_fit"]}
        b["step_mask"] = np.zeros((MAX_N, n_inst))
        for k, e in enumerate(s):
            b["exact_lik"][k] = EX_LIK[hyp, e - 1]
            b["gist_lik"][k] = GIST_LIK[hyp, e - 1]
            b["exact_fit"][k] = MEMBERSHIP[hyp, e - 1]
            b["gist_fit"][k] = GIST_FIT[hyp, e - 1]
            b["step_mask"][k] = 1.0
        b["inst_set"] = np.full(n_inst, i, dtype="int64")
        b["inst_is_rule"] = np.concatenate([np.ones(N_RULES), np.zeros(n_g)])
        b["inst_size"] = np.concatenate([np.ones(N_RULES), keys[:, 0].astype("float64")])
        b["inst_log_count"] = np.concatenate([np.zeros(N_RULES), np.log(count)])
        blocks.append(b)
        offsets.append(offset)
        members.append(np.concatenate([RULE_MEMBERSHIP, g_member], axis=0))  # (inst, 100)
        offset += n_inst
    out = {k: np.concatenate([b[k] for b in blocks], axis=-1) for k in blocks[0]}
    cells = list(dict.fromkeys((index[s], int(r["target"])) for s, r in zip(sets, rows)))
    cell_index = {c: j for j, c in enumerate(cells)}
    nz_inst, nz_cell, nz_share = [], [], []
    for j, (i, t) in enumerate(cells):
        share = members[i][:, t - 1]
        nz = np.nonzero(share)[0]
        nz_inst.append(offsets[i] + nz)
        nz_cell.append(np.full(len(nz), j, dtype="int64"))
        nz_share.append(share[nz])
    out.update({
        "set_ptr": np.array(offsets + [offset], dtype="int64"),
        "cell_dummy": np.zeros(len(cells)),
        "nz_inst": np.concatenate(nz_inst).astype("int64"),
        "nz_cell": np.concatenate(nz_cell).astype("int64"),
        "nz_share": np.concatenate(nz_share),
        "row_cell": np.array([cell_index[(index[s], int(r["target"]))] for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    })
    return out
