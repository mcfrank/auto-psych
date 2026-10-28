# name: target_in_set_yes_rate
# description: Yes rate on trials whose target is one of the shown examples; observed below null means people reject shown examples more often than the model (which treats examples as near-certain members) predicts.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    hit = np.any(ex == t[:, None], axis=1)
    return float(r[hit].mean()) if hit.any() else 0.0
