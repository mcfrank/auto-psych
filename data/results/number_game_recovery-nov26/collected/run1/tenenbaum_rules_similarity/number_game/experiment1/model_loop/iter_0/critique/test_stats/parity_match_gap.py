# name: parity_match_gap
# description: Among sets of >=2 examples all of one parity, yes rate for same-parity targets minus yes rate for opposite-parity targets; observed below the null mean means the model exaggerates how much humans generalise along parity.
def _parse(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    return ex, t, r

def test_statistic(df):
    ex, t, r = _parse(df)
    par = np.where(np.isnan(ex), np.nan, ex % 2)
    mx = np.nanmax(par, axis=1); mn = np.nanmin(par, axis=1)
    uni = (mx == mn) & ((~np.isnan(ex)).sum(axis=1) >= 2)
    same = uni & ((t % 2) == mn); diff = uni & ((t % 2) != mn)
    if not same.any() or not diff.any():
        return 0.0
    return float(r[same].mean() - r[diff].mean())
