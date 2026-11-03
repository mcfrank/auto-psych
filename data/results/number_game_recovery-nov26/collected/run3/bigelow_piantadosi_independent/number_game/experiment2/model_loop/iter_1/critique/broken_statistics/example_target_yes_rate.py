# name: example_target_yes_rate
# description: Yes rate on trials whose target is one of the shown examples; observed below the null mean means the model over-endorses the examples themselves (people reject a sizable share of examples).
def _ex(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    return E, t, r

def _rate(r, m):
    return float(r[m].mean()) if m.sum() > 0 else 0.0

def test_statistic(df):
    E, t, r = _ex(df)
    m = (E == t[:, None]).any(1)
    return _rate(r, m)
