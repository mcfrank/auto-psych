# name: alt_motifs_effect
# description: Covariance between the difference in the number of alternating motifs and the choice of the left sequence.
def test_statistic(df):
    diff_alt = df["alt_motifs_a"] - df["alt_motifs_b"]
    if len(df) <= 1:
        return 0.0
    cov = df["chose_left"].cov(diff_alt)
    import pandas as pd

    return float(cov) if pd.notna(cov) else 0.0
