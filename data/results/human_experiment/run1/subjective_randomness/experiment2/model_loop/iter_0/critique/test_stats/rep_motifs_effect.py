# name: rep_motifs_effect
# description: Covariance between the difference in the number of repeating motifs and the choice of the left sequence.
def test_statistic(df):
    diff_rep = df["rep_motifs_a"] - df["rep_motifs_b"]
    if len(df) <= 1:
        return 0.0
    cov = df["chose_left"].cov(diff_rep)
    import pandas as pd

    return float(cov) if pd.notna(cov) else 0.0
