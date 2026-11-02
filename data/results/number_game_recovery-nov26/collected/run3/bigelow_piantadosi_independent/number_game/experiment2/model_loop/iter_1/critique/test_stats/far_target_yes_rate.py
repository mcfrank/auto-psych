# name: far_target_yes_rate
# description: Yes rate for targets more than 30 away from every example; observed above null means people give more distant, rule-unrelated yeses (heavier tails/guessing) than the model's lapse allows.
def _ex(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    return E, t, r

def _rate(r, m):
    return float(r[m].mean()) if m.sum() > 0 else 0.0

def test_statistic(df):
    E, t, r = _ex(df)
    d = np.nanmin(np.abs(E - t[:, None]), 1)
    return _rate(r, d > 30)
