# name: above_max_minus_below_min
# description: Among non-example targets outside the example range by 1-15, yes rate for targets above the largest example minus targets below the smallest; observed above null means people extend generalisation upward more than downward (asymmetry the symmetric interval model lacks), below means the reverse.
def _ex(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    n = (~np.isnan(E)).sum(1)
    isex = (E == t[:, None]).any(1)
    return E, t, r, n, isex

def _rate(r, m):
    return float(r[m].mean()) if m.sum() > 0 else 0.0

def test_statistic(df):
    E, t, r, n, isex = _ex(df)
    mx = np.nanmax(E, 1); mn = np.nanmin(E, 1)
    up = (t > mx) & (t - mx <= 15)
    dn = (t < mn) & (mn - t <= 15)
    return _rate(r, up & ~isex) - _rate(r, dn & ~isex)
