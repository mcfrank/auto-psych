"""Evidence-tuned attention exemplar model.

People do not infer a concept; they store each example shown and judge a target
by its average similarity to those stored examples. Similarity to an example
falls off exponentially with the attention-weighted number of that example's
properties (parity, primality, squares/cubes, powers, multiples, base-ten
multiples, final digit, contained digits) the target lacks, plus their
magnitude distance. Attention to a property is learned from the examples: it is
the number of stored examples carrying the property divided by (number of
examples + a prior pseudo-count kappa), times a family salience; magnitude
attention is shrunk by the same factor n / (n + kappa). A lone example thus
draws weak attention (broad, nearly flat generalization); more examples sharpen
attention to the features they share.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

_N = list(range(1, 101))
N_FAM = 8


def _props():
    props = []
    props.append((0, {n for n in _N if n % 2 == 1}))
    props.append((0, {n for n in _N if n % 2 == 0}))
    props.append((1, {n for n in _N if n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))}))
    props.append((2, {k * k for k in range(1, 11)}))
    props.append((2, {k ** 3 for k in range(1, 5)}))
    for b in range(2, 11):
        props.append((3, {b ** k for k in range(1, 8) if b ** k <= 100}))
    for m in range(3, 13):
        props.append((7 if m in (5, 10) else 4, {n for n in _N if n % m == 0}))
    for d in range(10):
        props.append((5, {n for n in _N if n % 10 == d}))
    for d in range(10):
        props.append((6, {n for n in _N if str(d) in str(n)}))
    return props


_PROPS = _props()
_FAM = np.array([f for f, _ in _PROPS])
# F[number, prop] membership matrix, numbers 1..100 at rows 1..100
_F = np.zeros((101, len(_PROPS)))
for _j, (_f, _s) in enumerate(_PROPS):
    for _n in _s:
        _F[_n, _j] = 1.0
_FAM_ONEHOT = np.zeros((len(_PROPS), N_FAM))
_FAM_ONEHOT[np.arange(len(_PROPS)), _FAM] = 1.0


def prepare_observed(rows):
    n_tr = len(rows)
    mism = np.zeros((n_tr, 4, N_FAM))   # count-weighted missing features
    mag = np.zeros((n_tr, 4))
    mask = np.zeros((n_tr, 4))
    nex = np.zeros(n_tr)
    rating = np.zeros(n_tr, dtype="int64")
    cache = {}
    for i, r in enumerate(rows):
        s = str(r["number_set"])
        if s not in cache:
            ex = [int(x) for x in s.split(",")]
            cnt = _F[ex].sum(axis=0)  # how many examples carry each property
            cache[s] = (ex, cnt)
        ex, cnt = cache[s]
        t = int(r["target"])
        lack = 1.0 - _F[t]
        for k, e in enumerate(ex):
            mism[i, k] = (_F[e] * lack * cnt) @ _FAM_ONEHOT
            mag[i, k] = abs(t - e) / 10.0
            mask[i, k] = 1.0
        nex[i] = len(ex)
        rating[i] = int(float(r["rating"]))
    out = {"n_examples": nex, "ex_mask": mask, "mag_dist": mag, "rating": rating}
    for f in range(N_FAM):
        out[f"mism_{f}"] = mism[:, :, f]
    return out


with pm.Model() as model:
    nex = pm.Data("n_examples", np.ones(1, dtype="float64"))
    mask = pm.Data("ex_mask", np.ones((1, 4), dtype="float64"))
    mag = pm.Data("mag_dist", np.zeros((1, 4), dtype="float64"))
    mism = [pm.Data(f"mism_{f}", np.zeros((1, 4), dtype="float64")) for f in range(N_FAM)]

    salience = pm.HalfNormal("salience", sigma=1.0, shape=N_FAM)
    mag_att = pm.HalfNormal("mag_attention", sigma=0.5)
    kappa = pm.HalfNormal("kappa", sigma=2.0)
    intercept = pm.Normal("intercept", mu=-2.0, sigma=1.5)
    slope = pm.HalfNormal("slope", sigma=5.0)

    shrink = 1.0 / (nex[:, None] + kappa)          # attention = count / (n + kappa)
    feat_dist = sum(salience[f] * mism[f] for f in range(N_FAM))
    dist = shrink * feat_dist + (nex[:, None] * shrink) * mag_att * mag
    sim = pt.exp(-dist) * mask
    mean_sim = sim.sum(axis=1) / nex
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + slope * mean_sim))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
