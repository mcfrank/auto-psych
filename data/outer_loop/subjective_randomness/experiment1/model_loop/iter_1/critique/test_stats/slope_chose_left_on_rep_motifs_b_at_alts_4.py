# name: slope_chose_left_on_rep_motifs_b_at_alts_4
# description: The slope of chose_left on repeated motifs in sequence B restricted to alts_b == 4, isolating motif sensitivity from simple alternation rate.
def test_statistic(df):
    subset = df[df["alts_b"] == 4]
    if len(subset) < 2:
        return 0.0
    import numpy as np
    from scipy.stats import linregress

    slope, _, _, _, _ = linregress(subset["rep_motifs_b"], subset["chose_left"])
    return float(slope) if not np.isnan(slope) else 0.0
