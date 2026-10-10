"""Models the live experiments can tell apart (the 2026-10-09 rehearsal).

The live displays are plain: no valence, familiarization or greyscale. A
model term that reads only those is zero there, so models that differ only in
such terms make the same predictions on every display a live experiment can
show. The novelty gate let them through (its pool has valence and
familiarization displays), live-only selection could not separate them, and
experiment 3 of the rehearsal carried 12 near-copies into a design with
nothing to discriminate (power 0.25).

Two models are the same hypothesis *for the live phase* when their
posterior-mean choice-class probabilities on the design pool (every display
the EIG may pick, `src.rsa.design.run.design_pool`) are within
``SAME_ON_POOL_RMSE`` of each other. `keep_distinct` walks models in order
of preference and keeps each one unless it is the same as one already kept.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

import numpy as np

# The novelty gate's threshold, measured on the displays the live phase can
# show (fixed 2026-10-10 from fits to 28k human trials, dense mass): the
# rehearsal's carried twin_* models are a median 0.0002 apart; the three
# promoted singleton seeds that tie on plain displays (power analysis: each
# recovered a third of the time) 0.0018-0.0053; evaluative_prominence_l2 and
# rsa_l2, partly confused (0.6-0.8), 0.0024; rsa_l1 and rsa_l2, recovered at
# 0.86-0.90, 0.0070. A threshold of 0.01 would have merged rsa_l1 into rsa_l2.
# Rechecked on the wider live pool (1,597 displays, redundant features and two-object
# displays; 2026-10-10), same fits: distances grow 20-30% and keep their order
# (twins median 0.0002; the singleton seeds 0.0021-0.0068; evaluative_prominence_l2
# and rsa_l2 0.0031; rsa_l1 and rsa_l2 0.0086). 45 of 171 pairs fall below 0.002
# (46 before), so the threshold stays.
SAME_ON_POOL_RMSE = 0.002


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def keep_distinct(order: Sequence[str], preds: Mapping[str, np.ndarray], *, threshold: float = SAME_ON_POOL_RMSE,
                  cap: int = 0) -> Dict[str, List[dict]]:
    """Walk ``order`` (most preferred first); keep a model unless it is within
    ``threshold`` of one already kept, and stop keeping at ``cap`` (0: no cap).

    Returns ``kept`` (names), ``merged`` (each dropped twin with the kept model
    it duplicates and their RMSE) and ``over_cap`` (distinct models past the cap).
    """
    kept: List[str] = []
    merged, over_cap = [], []
    for name in order:
        dist = {k: rmse(preds[name], preds[k]) for k in kept}
        twin = min(dist, key=dist.get) if dist else None
        if twin is not None and dist[twin] < threshold:
            merged.append(dict(name=name, same_as=twin, rmse=dist[twin]))
        elif cap and len(kept) >= cap:
            over_cap.append(dict(name=name, nearest=twin, rmse=None if twin is None else dist[twin]))
        else:
            kept.append(name)
    return dict(kept=kept, merged=merged, over_cap=over_cap)
