# name: inside_range_nonrule_yes_rate
# description: Yes-rate for targets strictly inside [min,max] of 3-4 example sets, not equal to an example and sharing no last digit with any example; observed < null means the model over-predicts interval (filling-in) generalization, observed > null means people fill in ranges more than the model.
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
    share_digit = (valid & (np.nan_to_num(ex) % 10 == (t % 10)[:, None])).any(axis=1)
    m = (n >= 3) & (t > lo) & (t < hi) & ~share_digit
    return float(r[m].mean()) if m.any() else 0.0
