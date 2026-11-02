# name: trial_block_yes_variance
# description: Variance of the yes-rate across consecutive blocks of 50 trial_index positions (pooled over participants); observed above null means responding drifts over the session in a way the stationary model cannot produce.
def test_statistic(df):
    r = df["rating"].astype(float).to_numpy()
    b = (df["trial_index"].astype(int).to_numpy() // 50)
    s = pd.Series(r).groupby(b).mean()
    return float(s.var(ddof=0)) if len(s) > 1 else 0.0
