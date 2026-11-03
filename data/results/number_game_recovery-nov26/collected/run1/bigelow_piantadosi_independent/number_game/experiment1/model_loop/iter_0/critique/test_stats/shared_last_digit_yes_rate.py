# name: shared_last_digit_yes_rate
# description: Yes rate for non-example targets whose last digit equals the common last digit of a set of >=2 examples all ending in the same digit; observed < model means people endorse the ends-in-d rule less strongly than the model.
def _parse(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    lo = np.nanmin(ex, axis=1)
    hi = np.nanmax(ex, axis=1)
    par = np.where(np.isnan(ex), np.nan, ex % 2)
    all_even = np.nansum(par, axis=1) == 0
    all_odd = np.nansum(par, axis=1) == n
    return ex, t, y, n, dist, lo, hi, all_even, all_odd


def _rate(y, m):
    return float(y[m].mean()) if m.sum() > 0 else 0.0


def test_statistic(df):
    ex, t, y, n, dist, lo, hi, ae, ao = _parse(df)
    ld = ex % 10
    first = ld[:, 0]
    same = (n >= 2) & np.all(np.isnan(ld) | (ld == first[:, None]), axis=1)
    m = same & (t % 10 == first) & (dist > 0)
    return _rate(y, m)
