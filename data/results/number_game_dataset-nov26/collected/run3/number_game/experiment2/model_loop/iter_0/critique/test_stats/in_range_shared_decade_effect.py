# name: in_range_shared_decade_effect
# description: For non-example targets inside the min-max range of sets with 2+ examples, yes rate of targets sharing a tens digit with some example minus those sharing none; observed above null means people use decade (place-value) similarity more than the model's rule/interval hypotheses predict.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy()
    t = df["target"].astype(int).to_numpy().astype(float)
    n_ex = np.sum(~np.isnan(ex), axis=1)
    lo, hi = np.nanmin(ex, axis=1), np.nanmax(ex, axis=1)
    is_ex = (ex == t[:, None]).any(axis=1)
    tens_ex = np.floor(ex / 10)
    share = (tens_ex == np.floor(t / 10)[:, None]).any(axis=1)
    ok = (n_ex >= 2) & ~is_ex & (t > lo) & (t < hi)
    r = df["rating"].to_numpy().astype(float)
    a, b = ok & share, ok & ~share
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
