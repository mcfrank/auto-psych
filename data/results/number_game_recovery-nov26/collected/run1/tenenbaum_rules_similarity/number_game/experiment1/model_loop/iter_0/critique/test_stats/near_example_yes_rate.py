# name: near_example_yes_rate
# description: Yes rate for targets within 1-3 of some example but not equal to any example; observed below the null mean means the model over-produces "yes" for numerically adjacent targets (too much interval mass), above means it under-produces similarity-based yes.
def _parse(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    return ex, t, r

def test_statistic(df):
    ex, t, r = _parse(df)
    d = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    m = (d >= 1) & (d <= 3)
    return float(r[m].mean()) if m.any() else 0.0
