# name: yes_trial_index_slope
# description: OLS slope of yes (0/1) on trial_index (per 100 trials); observed more negative than null means people drift toward saying no over the session, which the stimulus-only model does not produce.
def _ex(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    return E, t, r

def _rate(r, m):
    return float(r[m].mean()) if m.sum() > 0 else 0.0

def test_statistic(df):
    x = df["trial_index"].astype(float).to_numpy() / 100.0
    r = df["rating"].astype(float).to_numpy()
    xc = x - x.mean()
    return float((xc * (r - r.mean())).sum() / max((xc ** 2).sum(), 1e-9))
