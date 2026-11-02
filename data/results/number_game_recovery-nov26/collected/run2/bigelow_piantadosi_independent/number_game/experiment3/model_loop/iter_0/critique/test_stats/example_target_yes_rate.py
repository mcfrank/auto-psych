# name: example_target_yes_rate
# description: Yes-rate on trials whose target is itself one of the shown examples; observed below null_mean means the model over-endorses the examples (it treats them as near-certain members while people say no to them ~14% of the time).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float)
    t = df["target"].astype(float).to_numpy()[:, None]
    inset = (ex.to_numpy() == t).any(axis=1)
    r = df["rating"].astype(float).to_numpy()
    return float(r[inset].mean()) if inset.any() else 0.0
