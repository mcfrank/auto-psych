# name: rep_motifs_aversion
# description: Correlation between the preference for sequence A and the difference in repeating motifs (B - A), for equal length sequences.
def test_statistic(df):
    import numpy as np

    mask = (df["n_a"] == df["n_b"]) & (df["rep_motifs_a"] != df["rep_motifs_b"])
    if mask.sum() == 0:
        return 0.0
    return float(
        np.corrcoef(
            df.loc[mask, "rep_motifs_b"] - df.loc[mask, "rep_motifs_a"],
            df.loc[mask, "chose_left"],
        )[0, 1]
    )
