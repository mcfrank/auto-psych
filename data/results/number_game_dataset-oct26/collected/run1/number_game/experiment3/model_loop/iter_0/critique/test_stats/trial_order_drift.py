# name: trial_order_drift
# description: Yes-rate on trials with trial_index above the median minus yes-rate at or below it; a nonzero observed value outside the null (which has no time dependence) means the model misses a drift in willingness to say yes over the session (positive = people get more liberal later).
def test_statistic(df):
    ti = df["trial_index"].to_numpy().astype(float)
    r = df["rating"].to_numpy().astype(float)
    med = np.median(ti)
    late = ti > med
    if late.sum() == 0 or (~late).sum() == 0:
        return 0.0
    return float(r[late].mean() - r[~late].mean())
