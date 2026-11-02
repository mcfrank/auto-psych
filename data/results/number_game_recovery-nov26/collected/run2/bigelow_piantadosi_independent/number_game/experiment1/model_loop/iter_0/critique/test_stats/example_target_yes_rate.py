# name: example_target_yes_rate
# description: Yes rate on trials where the target is itself one of the shown examples; t_observed below null_mean means people endorse shown examples less than the model (which predicts near-certain yes).
def test_statistic(df):
    s = df["number_set"].astype(str)
    t = df["target"].astype(int).astype(str)
    padded = "," + s.str.replace(" ", "", regex=False) + ","
    m = np.array([("," + b + ",") in a for a, b in zip(padded.values, t.values)], dtype=bool)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].values[m].mean())
