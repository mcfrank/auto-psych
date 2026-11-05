# name: four_example_above_minus_below_range
# description: For 4-example sets, yes-rate on targets above the max minus yes-rate on targets below the min; observed > model means a stronger upward asymmetry than the model predicts.
def _feat(df):
    s = df["number_set"].astype(str).str.split(",", expand=True)
    s = s.apply(lambda c: pd.to_numeric(c, errors="coerce"))
    a = s.to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(a), axis=1)
    mn = np.nanmin(a, axis=1)
    mx = np.nanmax(a, axis=1)
    is_ex = np.any(a == t[:, None], axis=1)
    par = np.where(np.isnan(a), np.nan, np.mod(a, 2))
    allodd = np.nansum(par == 1, axis=1) == n
    alleven = np.nansum(par == 0, axis=1) == n
    return t, y, n, mn, mx, is_ex, allodd, alleven

def _m(y, mask):
    return float(np.mean(y[mask])) if np.any(mask) else 0.0

def test_statistic(df):
    t, y, n, mn, mx, ex, ao, ae = _feat(df)
    k = n == 4
    return _m(y, k & (t > mx)) - _m(y, k & (t < mn))
