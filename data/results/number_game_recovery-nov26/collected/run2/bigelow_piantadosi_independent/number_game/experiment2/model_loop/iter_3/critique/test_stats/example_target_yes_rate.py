# name: example_target_yes_rate
# description: Yes rate on trials whose target is itself one of the shown examples; observed below null means people say no to shown examples (lapses/guessing) more than the model, which nearly always endorses them, predicts.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    pad = np.isnan(E)
    n = (~pad).sum(axis=1)
    in_set = (E == t[:, None]).any(axis=1)
    def rate(mask):
        return float(y[mask].mean()) if mask.any() else 0.0
    return rate(in_set)
