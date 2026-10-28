# name: far_outside_range_yes_rate
# description: Yes-rate for targets lying more than 10 below the smallest or above the largest example; observed above null means humans accept far out-of-range numbers more than the model (whose intervals cannot reach them) predicts.
def _ex(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values.astype(float)
    t = df["target"].astype(int).values.astype(float)
    r = df["rating"].astype(float).values
    return ex, t, r

def test_statistic(df):
    ex, t, r = _ex(df)
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    m = (t < lo - 10) | (t > hi + 10)
    return float(r[m].mean()) if m.any() else 0.0
