# name: mixed_parity_range_gap
# description: In sets of >=2 examples mixing odd and even numbers, yes rate for targets inside [min,max] of the examples minus targets outside it; observed above null means people use the example range more sharply than the model predicts.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    def rate(mask):
        return float(r[mask].mean()) if mask.sum() > 0 else 0.0
    par = np.where(np.isnan(ex), np.nan, ex % 2)
    mixed = (n >= 2) & (np.nanmax(par, axis=1) != np.nanmin(par, axis=1))
    lo, hi = np.nanmin(ex, axis=1), np.nanmax(ex, axis=1)
    inside = (t >= lo) & (t <= hi)
    return rate(mixed & inside) - rate(mixed & ~inside)
