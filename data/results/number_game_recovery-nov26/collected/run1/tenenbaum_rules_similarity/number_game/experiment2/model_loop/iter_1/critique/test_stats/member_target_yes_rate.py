# name: member_target_yes_rate
# description: Yes rate on trials whose target is one of the shown examples; observed below null means the model over-predicts endorsement of the examples themselves (too reliable), above means it under-predicts.
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",", expand=True).astype(float)
    t = df["target"].astype(float)
    m = s.eq(t, axis=0).any(axis=1).values
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].values[m].mean())
