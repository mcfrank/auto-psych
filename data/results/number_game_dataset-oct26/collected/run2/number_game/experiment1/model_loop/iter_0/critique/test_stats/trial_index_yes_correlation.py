# name: trial_index_yes_correlation
# description: Pearson correlation between trial_index and the yes response; the model has no time dependence, so observed away from null means yes-rate drifts across the session (positive = more yes later, negative = fewer).
def _ex(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values.astype(float)
    t = df["target"].astype(int).values.astype(float)
    r = df["rating"].astype(float).values
    return ex, t, r

def test_statistic(df):
    ex, t, r = _ex(df)
    ti = df["trial_index"].astype(float).values
    if ti.std() == 0 or r.std() == 0: return 0.0
    return float(np.corrcoef(ti, r)[0, 1])
