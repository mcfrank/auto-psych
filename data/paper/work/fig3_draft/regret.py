"""Which stimulus feature carries the discovered model's improvement over the best seed (scientific regret
minimization, Agrawal, Peterson & Griffiths, 2020, with the discovered model as the reference)?

Items are records with ``n`` (trials), ``observed``, ``best_seed`` and ``best_model`` (choice rates). A
feature is a function of an item's stimulus, fixed before looking (``FEATURES``). Each feature is scored by
how much better it lines up with the seed's errors than with the discovered model's:
|corr(feature, observed - seed)| - |corr(feature, observed - discovered)|, trial-weighted. The top feature
is the effect Figure 3's row C shows (row B is each domain's classic effect, fixed in advance).
"""
from typing import Callable, Dict, List

import numpy as np


def wcorr(x, y, w):
    w = w / w.sum()
    mx, my = (w * x).sum(), (w * y).sum()
    return (w * (x - mx) * (y - my)).sum() / np.sqrt((w * (x - mx) ** 2).sum() * (w * (y - my) ** 2).sum())


def rank_features(items: List[dict], stimulus: Callable[[dict], object],
                  features: Dict[str, Callable[[object], float]]) -> List[dict]:
    """Every feature with its correlation with each model's residuals and its score, best first. A feature
    constant over the items is skipped."""
    n = np.array([i["n"] for i in items], float)
    obs = np.array([i["observed"] for i in items])
    res_seed = obs - np.array([i["best_seed"] for i in items])
    res_best = obs - np.array([i["best_model"] for i in items])
    out = []
    for name, f in features.items():
        x = np.array([f(stimulus(i)) for i in items], float)
        if x.std() == 0:
            continue
        r_seed, r_best = wcorr(x, res_seed, n), wcorr(x, res_best, n)
        out.append({"feature": name, "r_seed": r_seed, "r_best": r_best, "score": abs(r_seed) - abs(r_best),
                    "binary": set(np.unique(x)) <= {0.0, 1.0}})
    return sorted(out, key=lambda d: -d["score"])


# --- subjective randomness: features of one coin sequence ("HHTHTT") ------------------------------

def _runs(s):
    out, c = [], 1
    for a, b in zip(s, s[1:]):
        if a == b:
            c += 1
        else:
            out.append(c)
            c = 1
    return out + [c]


def _periodic(s):
    """Made of one shorter block repeated (HTHTHT, HHTTHHTT)."""
    return any(len(s) % p == 0 and s == s[:p] * (len(s) // p) for p in range(1, len(s) // 2 + 1))


SR_FEATURES = {
    "alternation rate": lambda s: sum(a != b for a, b in zip(s, s[1:])) / (len(s) - 1),
    "heads/tails imbalance": lambda s: abs(s.count("H") - s.count("T")) / len(s),
    "longest run": lambda s: (max(_runs(s)) - 1) / (len(s) - 1),
    "final run": lambda s: (_runs(s)[-1] - 1) / (len(s) - 1),
    "first run": lambda s: (_runs(s)[0] - 1) / (len(s) - 1),
    "variety of run lengths": lambda s: len(set(_runs(s))) / len(_runs(s)),
    "palindrome": lambda s: float(s == s[::-1]),
    "mirror-complement": lambda s: float(s == s[::-1].translate(str.maketrans("HT", "TH"))),
    "periodic": lambda s: float(_periodic(s)),
    "length": lambda s: float(len(s)),
}
