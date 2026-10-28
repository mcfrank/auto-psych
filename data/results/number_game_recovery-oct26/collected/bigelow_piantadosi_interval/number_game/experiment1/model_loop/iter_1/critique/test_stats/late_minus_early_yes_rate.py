# name: late_minus_early_yes_rate
# description: Yes rate on trials in the second half of the session (trial_index >= median) minus the first half; nonzero observed vs a near-zero replicate mean means response bias drifts over the session (positive = growing yes-bias), which the stationary model cannot produce.
def test_statistic(df):
    ti = df["trial_index"].to_numpy().astype(float)
    r = df["rating"].to_numpy().astype(float)
    med = np.median(ti)
    late = ti >= med
    if late.sum() == 0 or (~late).sum() == 0:
        return 0.0
    return float(r[late].mean() - r[~late].mean())
