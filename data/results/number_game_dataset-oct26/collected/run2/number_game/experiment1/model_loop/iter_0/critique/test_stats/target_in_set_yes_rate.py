# name: target_in_set_yes_rate
# description: Yes-rate on trials where the target is itself one of the examples; observed below null means humans reject shown examples more often than the model's near-certain inclusion (the model over-produces yes here).
def _ex(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values.astype(float)
    t = df["target"].astype(int).values.astype(float)
    r = df["rating"].astype(float).values
    return ex, t, r

def test_statistic(df):
    ex, t, r = _ex(df)
    m = np.nansum(ex == t[:, None], axis=1) > 0
    return float(r[m].mean()) if m.any() else 0.0
