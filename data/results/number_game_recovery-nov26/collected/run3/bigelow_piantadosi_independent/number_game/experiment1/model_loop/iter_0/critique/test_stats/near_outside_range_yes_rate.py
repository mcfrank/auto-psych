# name: near_outside_range_yes_rate
# description: Yes-rate for targets outside [min,max] of the examples but within 10 of that range (excluding targets sharing a last digit with an example), for sets of 2+ examples; observed > null means people extend numerical intervals beyond the examples more than the model, observed < null means less.
def _parse(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    return ex, t

def test_statistic(df):
    ex, t = _parse(df)
    r = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(ex)
    n = valid.sum(axis=1)
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    gap = np.maximum(lo - t, t - hi)
    share_digit = (valid & (np.nan_to_num(ex) % 10 == (t % 10)[:, None])).any(axis=1)
    m = (n >= 2) & (gap > 0) & (gap <= 10) & ~share_digit
    return float(r[m].mean()) if m.any() else 0.0
