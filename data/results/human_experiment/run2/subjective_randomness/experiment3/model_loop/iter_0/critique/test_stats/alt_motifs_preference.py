# name: alt_motifs_preference
# description: Correlation between the preference for sequence A and the difference in alternating motifs (A - B), for equal length sequences.
def test_statistic(df):
    import numpy as np

    mask = (df["n_a"] == df["n_b"]) & (df["alt_motifs_a"] != df["alt_motifs_b"])
    if mask.sum() == 0:
        return 0.0
    return float(
        np.corrcoef(
            df.loc[mask, "alt_motifs_a"] - df.loc[mask, "alt_motifs_b"],
            df.loc[mask, "chose_left"],
        )[0, 1]
    )
