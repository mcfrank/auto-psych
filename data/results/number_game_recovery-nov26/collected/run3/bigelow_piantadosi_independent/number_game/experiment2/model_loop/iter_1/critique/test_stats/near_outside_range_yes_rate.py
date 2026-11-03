# name: near_outside_range_yes_rate
# description: Yes rate for targets lying just outside the examples' range (1-5 below the minimum or above the maximum); observed above null means people extend intervals past the examples more than the model's similarity/interval account predicts.
def _ex(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    return E, t, r

def _rate(r, m):
    return float(r[m].mean()) if m.sum() > 0 else 0.0

def test_statistic(df):
    E, t, r = _ex(df)
    lo = np.nanmin(E, 1); hi = np.nanmax(E, 1)
    d = np.where(t < lo, lo - t, np.where(t > hi, t - hi, 0))
    m = (d >= 1) & (d <= 5)
    return _rate(r, m)
