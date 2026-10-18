# name: yes_rate_target_in_set
# description: Proportion of "yes" when the target is one of the shown examples; observed below null means the model over-predicts certain endorsement of seen examples (humans are noisier/lapse), above means the model under-endorses them.
def _parse(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4))
    ex = ex.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    return ex, t, r

def test_statistic(df):
    ex, t, r = _parse(df)
    m = (ex == t[:, None]).any(axis=1)
    return float(r[m].mean()) if m.any() else 0.0
