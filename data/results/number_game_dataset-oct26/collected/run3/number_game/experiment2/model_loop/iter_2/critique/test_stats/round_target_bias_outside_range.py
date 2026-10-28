# name: round_target_bias_outside_range
# description: For targets outside the examples' min-max range, in sets with no example divisible by 10, yes-rate on targets divisible by 10 minus yes-rate on other targets; observed > model means people favour round (decade) numbers as targets more than the model predicts, observed < model means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    no_round = ~np.any(np.mod(ex, 10) == 0, axis=1)
    out = ((t < lo) | (t > hi)) & no_round
    r = out & (t % 10 == 0); o = out & (t % 10 != 0)
    if not r.any() or not o.any():
        return 0.0
    return float(y[r].mean() - y[o].mean())
