# name: example_target_yes_rate
# description: Yes rate on trials whose target is itself one of the shown examples; observed above null means the model under-endorses the examples themselves (too much noise/outside-concept mass), below means it over-endorses them.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric).values
    t = df["target"].astype(float).values
    m = (ex == t[:, None]).any(axis=1)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].values[m].mean())
