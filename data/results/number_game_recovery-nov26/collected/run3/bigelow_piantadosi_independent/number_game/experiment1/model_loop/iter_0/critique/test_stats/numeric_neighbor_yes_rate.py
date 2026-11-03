# name: numeric_neighbor_yes_rate
# description: Yes-rate for targets 1-5 away from the nearest example that share no last digit with any example, among sets whose examples span more than 20; observed > null means people use raw numerical similarity beyond what rules and a single interval predict, observed < null means less.
def _parse(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    return ex, t

def test_statistic(df):
    ex, t = _parse(df)
    r = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(ex)
    d = np.where(valid, np.abs(ex - t[:, None]), np.inf).min(axis=1)
    span = np.nanmax(ex, axis=1) - np.nanmin(ex, axis=1)
    share_digit = (valid & (np.nan_to_num(ex) % 10 == (t % 10)[:, None])).any(axis=1)
    m = (d >= 1) & (d <= 5) & ~share_digit & (span > 20)
    return float(r[m].mean()) if m.any() else 0.0
