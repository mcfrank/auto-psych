"""The number-game November final, neighbourhood_doubt_three_example_memory, with switchable mechanisms.

Run 2's winner, the best winner on all three runs pooled. Its features and
arithmetic are the final's (``data/results/number_game_dataset-nov26/collected/
run2/number_game/experiment3/model_loop/models/``), unchanged; only the
switches are new.

The structure, which has no parameters of its own: a target is judged by the
salience-weighted share of the examples' shared profile that it has
(``num / den``), through a logistic, with the cues' saliences on a simplex
(``salience``, one component per cue kept; with one cue, its salience is 1).

The mechanisms (``MECHANISMS``), every parameter in one of them:

* ``intercept``: the logistic's offset (off: 0); ``slope``: its gain on the
  share (off: 1).
* the cues (``CUES``), each one component of ``salience``:
  - the property families the examples share: ``parity``, ``prime``,
    ``square_cube`` (squares and cubes), ``power`` (powers of 2-10),
    ``multiple`` (multiples of 3-12 but 5 and 10), ``last_digit``,
    ``contains_digit``, ``round_multiple`` (multiples of 5 and 10);
  - ``range_low``, ``range_high``: the target not below the smallest example,
    not above the largest, fading as a Gaussian of the gap (their shared
    ``band_width``). One of them stays (``KEEP_ONE_OF``): every item has them in
    the denominator, which no other cue guarantees;
  - ``seed``: multiples of the smallest example, when every example is one;
  - ``progression``: the examples' arithmetic progression, continued;
  - ``span``: closeness to the examples' span (``span_width``).
* ``retain``: a visible shared last digit down-weights every other cue to
  ``retain`` (off: 1); ``recency`` / ``primacy``: when the examples share no
  last digit, the last / first example's last digit, weighted relative to the
  last-digit cue. All three act through the last-digit cue's salience.
* the rule-ambiguity doubt: with a probability that rises with the number of
  rules every example fits, the target is judged by a flat "yes" rate
  (``random_yes``) instead. ``doubt_single`` / ``doubt_multi``: that doubt on
  one-example / multi-example sets (off: none there; ``random_yes`` stays
  while either is on); ``doubt_rich`` / ``doubt_rich_multi``: its rise with the
  number of rules (off: a constant rate); ``parity_keep``: the flat "yes"
  keeps the examples' shared parity; ``doubt_band``: on multi-example sets it
  keeps their magnitude neighbourhood too (the change run 2 ended on).
* ``memory_limit``: with four examples, one drops out of working memory with
  probability ``drop`` and the judgment averages over which (off: never).

A mechanism's neutral value (``NEUTRAL``) gives exactly the model without it:
a cue's own component of ``salience`` at 0 (the judgment is a ratio, so the
others' scale does not matter), any other mechanism's parameters as listed.
"""
import itertools

import numpy as np
import pymc as pm
import pytensor.tensor as pt

FINAL = "neighbourhood_doubt_three_example_memory"
NSUB = 5  # full set, then the four 3-example subsets (copies for n < 4)

_N = range(1, 101)
N_FAM = 8
FD = 5

FAMILIES = ("parity", "prime", "square_cube", "power", "multiple", "last_digit", "contains_digit",
            "round_multiple")  # the final's families 0-7, in its order
RANGE = ("range_low", "range_high")
CUES = (*FAMILIES, *RANGE, "seed", "progression", "span")  # the final's salience components, in its order
MECHANISMS = (
    "intercept", "slope", *CUES, "retain", "recency", "primacy",
    "doubt_single", "doubt_rich", "doubt_multi", "doubt_rich_multi", "parity_keep", "doubt_band",
    "memory_limit",
)
REQUIRES = {
    "retain": ("last_digit",),
    "recency": ("last_digit",),
    "primacy": ("last_digit",),
    "doubt_rich": ("doubt_single",),
    "doubt_rich_multi": ("doubt_multi",),
    "parity_keep": ("doubt_single", "doubt_multi"),
    "doubt_band": ("doubt_multi",),
}
KEEP_ONE_OF = (RANGE,)
# For variant names (``variants.variant_name``).
SHORT = {
    "intercept": "int", "slope": "slp", "parity": "par", "prime": "prime", "square_cube": "sq", "power": "pow", "multiple": "mult",
    "last_digit": "ld", "contains_digit": "cd", "round_multiple": "round", "range_low": "lo",
    "range_high": "hi", "seed": "seed", "progression": "prog", "span": "span", "retain": "ret",
    "recency": "rec", "primacy": "prim", "doubt_single": "d1", "doubt_rich": "dr1", "doubt_multi": "dm",
    "doubt_rich_multi": "drm", "parity_keep": "pk", "doubt_band": "dband", "memory_limit": "mem",
}
NEUTRAL = {
    "intercept": {"intercept": 0.0},
    "slope": {"slope": 1.0},
    **{cue: {"salience": "this cue's component at 0"} for cue in CUES},
    "retain": {"retain": 1.0},
    "recency": {"recency": 0.0},
    "primacy": {"primacy": 0.0},
    # A doubt rate of 0: its logit so low that the sigmoid is 0 in float64.
    "doubt_single": {"doubt0": -1e3},
    "doubt_rich": {"doubt_rich": 0.0},
    "doubt_multi": {"doubt_multi": -1e3},
    "doubt_rich_multi": {"doubt_rich_multi": 0.0},
    "parity_keep": {"parity_keep": 0.0},
    "doubt_band": {"doubt_band_width": np.inf},
    "memory_limit": {"drop": 0.0},
}


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
        fam = 7 if m in (5, 10) else 4
        props.append((fam, {n for n in _N if n % m == 0}))
    for d in range(10):
        props.append((5, {n for n in _N if n % 10 == d}))
    for d in range(10):
        props.append((6, {n for n in _N if str(d) in str(n)}))
    return props


_PROPS = _props()


def _feat(ex, t):
    out = {}
    for f in range(N_FAM):
        shared = [p for fam, p in _PROPS if fam == f and all(e in p for e in ex)]
        out[f"sh_{f}"] = float(len(shared))
        out[f"hit_{f}"] = float(sum(1 for p in shared if t in p))
    span_on = len(ex) >= 2
    out["single_example"] = 0.0 if span_on else 1.0
    # Number of candidate rules (property categories) every example fits,
    # centred at 6; for a lone example this is its own category count.
    out["rule_count"] = float(sum(1 for fam, p in _PROPS if all(e in p for e in ex))) - 6.0
    out["span_active"] = 1.0 if span_on else 0.0
    # Doubt-mode parity constraint: examples all share a parity, target breaks it.
    pars = {e % 2 for e in ex}
    out["shared_parity_break"] = 1.0 if (len(pars) == 1 and t % 2 not in pars) else 0.0
    out["span_gap"] = (max(0, min(ex) - t, t - max(ex)) / 100.0) if span_on else 0.0
    out["below_gap"] = max(0, min(ex) - t) / 100.0
    out["above_gap"] = max(0, t - max(ex)) / 100.0
    shared_digit = len({e % 10 for e in ex}) == 1
    out["visible_digit_pattern"] = 1.0 if (len(ex) >= 2 and shared_digit) else 0.0
    seed = min(ex)
    is_seed = seed >= 2 and all(e % seed == 0 for e in ex)
    out["seed_shared"] = 1.0 if is_seed else 0.0
    out["seed_hit"] = 1.0 if (is_seed and t % seed == 0) else 0.0
    srt = sorted(ex)
    gap = srt[1] - srt[0] if len(srt) >= 2 else 0
    is_prog = gap > 0 and all(srt[i + 1] - srt[i] == gap for i in range(len(srt) - 1))
    out["progression_shared"] = 1.0 if is_prog else 0.0
    out["progression_hit"] = 1.0 if (is_prog and t >= srt[0] and (t - srt[0]) % gap == 0) else 0.0
    linger = len(ex) >= 2 and not shared_digit
    out["last_digit_active"] = 1.0 if linger else 0.0
    out["last_digit_match"] = 1.0 if (linger and t % 10 == ex[-1] % 10) else 0.0
    out["first_digit_match"] = 1.0 if (linger and t % 10 == ex[0] % 10) else 0.0
    return out


KEYS = ["single_example", "rule_count", "span_active", "shared_parity_break", "span_gap",
        "below_gap", "above_gap", "visible_digit_pattern", "seed_shared", "seed_hit",
        "progression_shared", "progression_hit", "last_digit_active", "last_digit_match",
        "first_digit_match"] + [f"sh_{f}" for f in range(N_FAM)] + [f"hit_{f}" for f in range(N_FAM)]


def prepare_observed(rows):
    """The features once per distinct (example set, target) item, in order of first
    appearance, and each response's item (``item``): P(yes) is computed per item, about
    ten times fewer than the responses, and the likelihood is still per response. The
    final computes the same features per response."""
    items = {}
    item = np.zeros(len(rows), dtype="int64")
    rating = np.zeros(len(rows), dtype="int64")
    for i, r in enumerate(rows):
        item[i] = items.setdefault((str(r["number_set"]), str(r["target"])), len(items))
        rating[i] = int(float(r["rating"]))
    out = {k: np.zeros((len(items), NSUB)) for k in KEYS}
    out["over_capacity"] = np.zeros(len(items))
    for j, (number_set, target) in enumerate(items):
        ex = [int(x) for x in number_set.split(",")]
        t = int(target)
        if len(ex) >= 4:
            subs = [ex] + [list(c) for c in itertools.combinations(ex, 3)]
        else:
            subs = [ex] * NSUB
        fs = [_feat(s, t) for s in subs]
        for k in KEYS:
            out[k][j] = [f[k] for f in fs]
        out["over_capacity"][j] = 1.0 if len(ex) >= 4 else 0.0
    out["item"] = item
    out["rating"] = rating
    return out


def _total(terms):
    """The sum of the terms kept (``None`` marks one switched off), in their order."""
    kept = [t for t in terms if t is not None]
    total = kept[0]
    for t in kept[1:]:
        total = total + t
    return total


def build(on):
    on = frozenset(on)
    unknown = sorted(on - set(MECHANISMS))
    if unknown:
        raise ValueError(f"unknown mechanisms {unknown}; known: {list(MECHANISMS)}")
    cues = [c for c in CUES if c in on]

    with pm.Model() as model:
        D = {k: pm.Data(k, np.zeros((1, NSUB))) for k in KEYS}
        oc = pm.Data("over_capacity", np.zeros(1))
        sh = [D[f"sh_{f}"] for f in range(N_FAM)]
        hit = [D[f"hit_{f}"] for f in range(N_FAM)]

        intercept = pm.Normal("intercept", mu=-2.0, sigma=1.5) if "intercept" in on else 0.0
        slope = pm.HalfNormal("slope", sigma=5.0) if "slope" in on else 1.0
        if len(cues) > 1:
            salience_kept = pm.Dirichlet("salience", a=np.full(len(cues), 1.5))
            salience = {c: salience_kept[i] for i, c in enumerate(cues)}
        else:  # one cue: its salience is the whole simplex, 1
            salience = {cues[0]: 1.0}
        retain = pm.Beta("retain", alpha=2.0, beta=2.0) if "retain" in on else None
        recency = pm.HalfNormal("recency", sigma=1.0) if "recency" in on else None
        primacy = pm.HalfNormal("primacy", sigma=1.0) if "primacy" in on else None
        if on & set(RANGE):
            band_width = pm.LogNormal("band_width", mu=np.log(0.25), sigma=0.5)
            width = 0.02 + band_width
            ge_min = pt.exp(-((D["below_gap"] / width) ** 2))
            le_max = pt.exp(-((D["above_gap"] / width) ** 2))
        span_w = pm.LogNormal("span_width", mu=np.log(0.05), sigma=0.5) if "span" in on else None

        other_w = 1.0 - (1.0 - retain) * D["visible_digit_pattern"] if retain is not None else 1.0
        num = other_w * _total([
            salience["range_low"] * ge_min if "range_low" in on else None,
            salience["range_high"] * le_max if "range_high" in on else None,
            salience["seed"] * D["seed_hit"] if "seed" in on else None,
        ])
        den = other_w * _total([
            salience["range_low"] if "range_low" in on else None,
            salience["range_high"] if "range_high" in on else None,
            salience["seed"] * D["seed_shared"] if "seed" in on else None,
        ])
        for f, family in enumerate(FAMILIES):
            if family not in on:
                continue
            w = salience[family] if f == FD else salience[family] * other_w
            num = num + w * hit[f]
            den = den + w * sh[f]
        if recency is not None:
            w_last = recency * salience["last_digit"]
            num = num + w_last * D["last_digit_match"]
            den = den + w_last * D["last_digit_active"]
        if "progression" in on:
            w_prog = other_w * salience["progression"]
            num = num + w_prog * D["progression_hit"]
            den = den + w_prog * D["progression_shared"]
        if primacy is not None:
            w_first = primacy * salience["last_digit"]
            num = num + w_first * D["first_digit_match"]
            den = den + w_first * D["last_digit_active"]
        if span_w is not None:
            span_sim = D["span_active"] * pt.exp(-(D["span_gap"] / (0.01 + span_w)))
            w_span = other_w * salience["span"]
            num = num + w_span * span_sim
            den = den + w_span * D["span_active"]
        p_judge = pm.math.sigmoid(intercept + slope * num / den)

        if on & {"doubt_single", "doubt_multi"}:
            d0 = pm.Normal("doubt0", 0.0, 1.5) if "doubt_single" in on else None
            db = pm.HalfNormal("doubt_rich", 1.0) if "doubt_rich" in on else 0.0
            dm = pm.Normal("doubt_multi", -2.0, 1.5) if "doubt_multi" in on else None
            dbm = pm.HalfNormal("doubt_rich_multi", 1.0) if "doubt_rich_multi" in on else 0.0
            ry = pm.Beta("random_yes", 2.0, 2.0)
            parity_keep = pm.Beta("parity_keep", alpha=2.0, beta=2.0) if "parity_keep" in on else 0.0
            single = D["single_example"]
            rich = D["rule_count"]
            if d0 is not None and dm is not None:
                doubt_t = pm.math.sigmoid(single * (d0 + db * rich) + (1.0 - single) * (dm + dbm * rich))
            elif d0 is not None:  # no doubt on multi-example sets
                doubt_t = single * pm.math.sigmoid(d0 + db * rich)
            else:  # no doubt on one-example sets
                doubt_t = (1.0 - single) * pm.math.sigmoid(dm + dbm * rich)
            if "doubt_band" in on:
                doubt_w = pm.LogNormal("doubt_band_width", mu=np.log(0.3), sigma=0.7)
                mag_keep = pt.exp(-((D["span_gap"] / (0.02 + doubt_w)) ** 2))
            else:
                mag_keep = 1.0
            p_doubt = (ry * (1.0 - parity_keep * D["shared_parity_break"])
                       * (single + (1.0 - single) * mag_keep))
            p_ver = doubt_t * p_doubt + (1.0 - doubt_t) * p_judge  # (trials, NSUB)
        else:
            p_ver = p_judge

        if "memory_limit" in on:
            drop = pm.Beta("drop", 2.0, 2.0)
            p_mix = (1.0 - drop * oc) * p_ver[:, 0] + drop * oc * pt.mean(p_ver[:, 1:], axis=1)
        else:
            p_mix = p_ver[:, 0]
        item = pm.Data("item", np.zeros(1, dtype="int64"))
        p_yes = pm.Deterministic("p_yes", pt.clip(p_mix, 1e-6, 1 - 1e-6)[item])

        rating = pm.Data("rating", np.zeros(1, dtype="int64"))
        pm.Bernoulli("response", p=p_yes, observed=rating)
    return model
