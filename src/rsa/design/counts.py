"""Collected rows back to the design: response counts per designed display and choice class.

The page randomises everything a model does not see (the item, which of its
features plays each abstract column, the screen order, the base images), and
`src.rsa.experiment.convert` undoes it: each row keeps the design's
canonical object order and column order, the chosen object's canonical index
and, in its covariates, the design trial it came from (``spec_index``) and the
design's hash. This module closes the loop to the design's terms: for each
designed display, how many people chose each choice class, in the
(displays, WIDTH) layout of `src.rsa.design.eig` (a class's count at its first
object's index).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.rsa.dataset import context_from_row
from src.rsa.experiment.design import Design

WIDTH = 4


def display_counts(rows: pd.DataFrame, design: Design) -> np.ndarray:
    """(len(design.specs), WIDTH) counts of each choice class per designed display.

    Catch trials are skipped. Raises when a row is not from this design (its
    hash), or when its display is not the designed one (objects, word): a
    bookkeeping error, never something to count past.
    """
    counts = np.zeros((len(design.specs), WIDTH), dtype=int)
    for _, row in rows.iterrows():
        cov = json.loads(row["covariates"])
        if cov.get("is_catch"):
            continue
        if cov["design_sha256"] != design.sha256:
            raise ValueError(f"row of design {cov['design_sha256'][:12]}, not {design.sha256[:12]}")
        spec = design.specs[int(cov["spec_index"])]
        ctx = context_from_row(row)
        if ctx.objects != spec.objects or ctx.utterance != spec.utterance:
            raise ValueError(f"row shows {ctx.objects} / word {ctx.utterance}, its design trial "
                             f"{cov['spec_index']} is {spec.objects} / word {spec.utterance}")
        counts[int(cov["spec_index"]), ctx.choice_classes()[int(row["choice"])]] += 1
    return counts
