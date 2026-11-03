# name: round_number_target_bias
# description: In sets containing no multiple of 10, yes rate for targets that are multiples of 10 minus yes rate for other targets; observed above null means people favour round-number targets beyond what the model's rules/intervals imply, below means people disfavour them relative to the model.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    lo = np.nanmin(ex, axis=1)
    hi = np.nanmax(ex, axis=1)
    no10 = ~np.any(np.nan_to_num(ex, nan=1.0) % 10 == 0, axis=1)
    r = (t % 10 == 0)
    a, b = no10 & r, no10 & ~r
    if not a.any() or not b.any():
        return 0.0
    return float(y[a].mean() - y[b].mean())
