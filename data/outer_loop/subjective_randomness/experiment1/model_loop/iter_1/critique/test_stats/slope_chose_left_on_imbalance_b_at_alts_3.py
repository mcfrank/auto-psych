# name: slope_chose_left_on_imbalance_b_at_alts_3
# description: The slope of chose_left on imbalance in sequence B restricted to alts_b == 3, isolating bias sensitivity from alternation rate.
def test_statistic(df):
    subset = df[df["alts_b"] == 3]
    if len(subset) < 2:
        return 0.0
    import numpy as np
    from scipy.stats import linregress

    slope, _, _, _, _ = linregress(subset["imbalance_b"], subset["chose_left"])
    return float(slope) if not np.isnan(slope) else 0.0
