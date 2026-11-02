# name: parity_mismatch_yes_rate
# description: Yes rate for targets whose parity differs from that of every example, among sets whose examples all share one parity; observed above the null mean means the model under-produces "yes" for parity-violating targets (too sharp an odd/even rule).
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
    m = uni & ((t % 2) != mn)
    return float(r[m].mean()) if m.any() else 0.0
