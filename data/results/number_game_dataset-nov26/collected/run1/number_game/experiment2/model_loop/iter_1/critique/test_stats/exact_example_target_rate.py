# name: exact_example_target_rate
# description: Yes-rate on trials whose target equals one of the set's examples; observed < null means people endorse the shown examples less than the model's maximal self-similarity predicts (a response-noise ceiling the model lacks), observed > null means more.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    uniq = sets.unique()
    exs = sets.map({s: frozenset(int(v) for v in s.split(",")) for s in uniq})
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    isex = np.array([ti in e for ti, e in zip(t, exs)])
    if isex.sum() == 0:
        return 0.0
    return float(r[isex].mean())
