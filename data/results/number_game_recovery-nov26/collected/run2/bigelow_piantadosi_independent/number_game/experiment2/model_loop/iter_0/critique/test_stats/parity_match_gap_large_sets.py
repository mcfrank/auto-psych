# name: parity_match_gap_large_sets
# description: In sets of >=3 examples that are all even or all odd, yes rate for targets of matching parity minus mismatching parity; observed above null means people rely on parity more than the model, below means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    def rate(mask):
        return float(r[mask].mean()) if mask.sum() > 0 else 0.0
    par = np.where(np.isnan(ex), np.nan, ex % 2)
    same = (n >= 3) & (np.nanmax(par, axis=1) == np.nanmin(par, axis=1))
    p0 = np.nanmin(par, axis=1)
    match = (t % 2) == p0
    return rate(same & match) - rate(same & ~match)
