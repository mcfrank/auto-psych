# name: extreme_max_run_rejection
# description: Mean probability of choosing left when sequence A has a max_run >= 4 and sequence B has max_run < 3.

import numpy as np


def test_statistic(df):
    mask = (df["max_run_a"] >= 4) & (df["max_run_b"] < 3)
    if mask.sum() == 0:
        return 0.0
    return float(df["chose_left"][mask].mean())
