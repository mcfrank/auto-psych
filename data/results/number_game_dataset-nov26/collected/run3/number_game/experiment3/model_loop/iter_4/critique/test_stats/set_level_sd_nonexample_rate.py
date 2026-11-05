# name: set_level_sd_nonexample_rate
# description: Standard deviation across number sets of the per-set yes-rate on non-example targets; observed > model means sets differ more sharply in breadth of generalization than the model produces.
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
    k = ~ex
    if not np.any(k):
        return 0.0
    g = pd.Series(y[k]).groupby(df["number_set"].astype(str).to_numpy()[k]).mean()
    return float(g.std(ddof=0)) if len(g) > 1 else 0.0
