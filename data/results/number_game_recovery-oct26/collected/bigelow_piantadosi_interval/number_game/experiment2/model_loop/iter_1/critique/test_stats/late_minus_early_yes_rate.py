# name: late_minus_early_yes_rate
# description: Yes-rate in the second half of each session (trial_index >= 225) minus the first half; a positive/negative observed value beyond the null means people drift toward more/fewer yeses over the session, a drift the stationary model does not produce.
def test_statistic(df):
    r = df["rating"].astype(float).to_numpy()
    ti = df["trial_index"].astype(int).to_numpy()
    late = ti >= 225
    if late.sum() == 0 or (~late).sum() == 0:
        return 0.0
    return float(r[late].mean() - r[~late].mean())
