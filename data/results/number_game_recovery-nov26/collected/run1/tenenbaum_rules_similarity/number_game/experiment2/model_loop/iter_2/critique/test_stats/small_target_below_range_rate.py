# name: small_target_below_range_rate
# description: Yes rate for targets <= 20 that lie below the smallest example (and are not members); observed above null means people have a small-number bias / extend downward toward 1 more than the model, below null means less.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    lo = np.nanmin(ex, axis=1)
    m = (t <= 20) & (t < lo)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy()[m].mean())
