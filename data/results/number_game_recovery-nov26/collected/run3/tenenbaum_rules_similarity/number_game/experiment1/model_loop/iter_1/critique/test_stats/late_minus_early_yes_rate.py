# name: late_minus_early_yes_rate
# description: Yes-rate on trials with trial_index >= 225 minus yes-rate on earlier trials; a nonzero observed value vs model (which has no time dependence) means people become more (positive) or less (negative) liberal over the session.
def test_statistic(df):
    t = df["trial_index"].to_numpy()
    r = df["rating"].to_numpy().astype(float)
    late = t >= 225
    if late.sum() == 0 or (~late).sum() == 0:
        return 0.0
    return float(r[late].mean() - r[~late].mean())
