# name: late_minus_early_yes_rate
# description: Yes-rate on trials with trial_index above the median minus yes-rate at or below it; the model has no session dynamics, so observed above null means people become more liberal (more yes) later in the session, below means more conservative.
def test_statistic(df):
    t = df["trial_index"].to_numpy(float)
    y = df["rating"].to_numpy(float)
    med = np.median(t)
    late = t > med
    if late.all() or (~late).all():
        return 0.0
    return float(y[late].mean() - y[~late].mean())
