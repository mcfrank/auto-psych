# name: high_periodicity_rejection
# description: Mean probability of choosing left when sequence A is highly periodic (>0.6) but sequence B is not (<0.4).

import numpy as np


def test_statistic(df):
    mask = (df["periodicity_a"] > 0.6) & (df["periodicity_b"] < 0.4)
    if mask.sum() == 0:
        return 0.0
    return float(df["chose_left"][mask].mean())
