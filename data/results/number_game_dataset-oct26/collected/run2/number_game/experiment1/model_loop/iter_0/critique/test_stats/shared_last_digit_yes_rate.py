# name: shared_last_digit_yes_rate
# description: Yes-rate for non-example targets sharing their final digit with at least one example and lying more than 5 from every example; observed above null means humans use surface digit similarity more than the model's final-digit rules imply, below null less.
def _ex(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values.astype(float)
    t = df["target"].astype(int).values.astype(float)
    r = df["rating"].astype(float).values
    return ex, t, r

def test_statistic(df):
    ex, t, r = _ex(df)
    d = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    share = np.nansum(np.mod(ex, 10) == np.mod(t, 10)[:, None], axis=1) > 0
    m = share & (d > 5)
    return float(r[m].mean()) if m.any() else 0.0
