# name: far_unrelated_yes_rate
# description: Yes-rate for targets >15 from every example, sharing no last digit with any example, and of a different parity than at least one example; observed > null means people say yes to unrelated targets more than the model (missing lapse/guessing), observed < null means the model over-generalizes.
def _parse(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    return ex, t

def test_statistic(df):
    ex, t = _parse(df)
    r = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(ex)
    d = np.where(valid, np.abs(ex - t[:, None]), np.inf).min(axis=1)
    share_digit = (valid & (np.nan_to_num(ex) % 10 == (t % 10)[:, None])).any(axis=1)
    par_diff = (valid & (np.nan_to_num(ex) % 2 != (t % 2)[:, None])).any(axis=1)
    m = (d > 15) & ~share_digit & par_diff
    return float(r[m].mean()) if m.any() else 0.0
