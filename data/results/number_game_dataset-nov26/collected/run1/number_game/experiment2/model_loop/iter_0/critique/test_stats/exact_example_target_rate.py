# name: exact_example_target_rate
# description: Yes-rate on trials whose target is identical to one of the set's examples; observed < null means people endorse the shown examples themselves less often than the model's maximal self-similarity predicts (i.e. a ceiling / response-noise floor the model lacks), observed > null means more.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    m = (ex == t[:, None]).any(1)
    r = df["rating"].to_numpy()
    return float(r[m].mean()) if m.sum() else 0.0
