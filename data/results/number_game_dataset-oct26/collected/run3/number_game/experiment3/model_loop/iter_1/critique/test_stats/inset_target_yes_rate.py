# name: inset_target_yes_rate
# description: Yes-rate on trials whose target is itself one of the shown examples; observed < null means people deny shown examples more often than the model (which treats examples as near-certain members) predicts, observed > null means the model under-endorses them.
def test_statistic(df):
    codes, uniq = pd.factorize(df["number_set"].astype(str))
    E = pd.Series(uniq).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)[codes]
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    m = np.any(E == t[:, None], axis=1)
    return float(y[m].mean()) if m.any() else 0.0
