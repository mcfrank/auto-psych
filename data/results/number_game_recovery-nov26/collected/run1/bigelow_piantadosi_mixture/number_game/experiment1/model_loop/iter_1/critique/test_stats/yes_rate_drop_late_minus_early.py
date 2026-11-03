# name: yes_rate_drop_late_minus_early
# description: Yes rate on the last fifth of trials (trial_index >= 360) minus the yes rate on the first fifth (trial_index < 90); a more negative observed value than the null means people grow more conservative over the session, which the stationary model fails to produce.
def test_statistic(df):
    t = df["trial_index"].astype(int).values
    r = df["rating"].astype(float).values
    late, early = r[t >= 360], r[t < 90]
    if len(late) == 0 or len(early) == 0:
        return 0.0
    return float(late.mean() - early.mean())
