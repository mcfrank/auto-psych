# name: example_as_target_yes_rate
# description: Yes rate on trials where the target is one of the set's own examples; observed below null means people reject shown examples more often than the model (which treats examples as near-certain members) predicts.
def test_statistic(df):
    parts = df["number_set"].astype(str).str.split(",", expand=True)
    ex = parts.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()[:, None]
    hit = (ex == t).any(axis=1)
    if hit.sum() == 0:
        return 0.0
    return float(df["rating"].astype(float).to_numpy()[hit].mean())
