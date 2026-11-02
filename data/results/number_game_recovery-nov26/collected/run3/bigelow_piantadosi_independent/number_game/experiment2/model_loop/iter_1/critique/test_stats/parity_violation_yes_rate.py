# name: parity_violation_yes_rate
# description: Among sets of 2+ examples that are all even or all odd, yes rate for targets of the opposite parity; observed above null means people commit to parity-consistent rules less than the model (model too all-or-none), below means more.
def _ex(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    return E, t, r

def _rate(r, m):
    return float(r[m].mean()) if m.sum() > 0 else 0.0

def test_statistic(df):
    E, t, r = _ex(df)
    n = (~np.isnan(E)).sum(1)
    par = np.where(np.isnan(E), np.nan, E % 2)
    s = np.nansum(par, 1)
    alleven = s == 0; allodd = s == n
    m = (n >= 2) & ((alleven & (t % 2 == 1)) | (allodd & (t % 2 == 0)))
    return _rate(r, m)
