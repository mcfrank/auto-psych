# name: example_target_yes_rate
# description: Yes rate on trials whose target is itself one of the shown examples; observed below null means people endorse the examples themselves less than the model (which treats examples as near-certain members) predicts.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int).to_numpy()
    is_ex = (ex.to_numpy() == t[:, None]).any(axis=1)
    if is_ex.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy()[is_ex].mean())
