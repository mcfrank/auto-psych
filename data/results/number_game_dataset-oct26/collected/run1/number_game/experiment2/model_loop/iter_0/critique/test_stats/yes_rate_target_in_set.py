# name: yes_rate_target_in_set
# description: Yes-rate among trials whose target is literally one of the examples; observed above model mean means the model under-produces near-certain yes for shown examples.
def test_statistic(df):
    X = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(X)
    n = valid.sum(axis=1)
    m = (X == t[:, None]).any(axis=1)
    return float(r[m].mean()) if m.any() else 0.0
