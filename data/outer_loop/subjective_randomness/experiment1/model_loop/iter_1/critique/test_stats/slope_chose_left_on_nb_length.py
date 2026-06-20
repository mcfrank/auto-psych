# name: slope_chose_left_on_nb_length
# description: The slope of chose_left on the length of sequence B.
def test_statistic(df):
    import numpy as np
    from scipy.stats import linregress

    slope, _, _, _, _ = linregress(df["n_b"], df["chose_left"])
    return float(slope) if not np.isnan(slope) else 0.0
