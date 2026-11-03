# name: set_size_yes_slope
# description: Yes rate on non-member targets for 1-example sets minus that for 4-example sets; observed below null means the model's size principle over-predicts how much extra examples narrow generalisation, above means it under-predicts.
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",", expand=True).astype(float)
    t = df["target"].astype(float)
    k = s.notna().sum(axis=1).values
    member = s.eq(t, axis=0).any(axis=1).values
    r = df["rating"].values.astype(float)
    a, b = (k == 1) & ~member, (k == 4) & ~member
    if not a.any() or not b.any():
        return 0.0
    return float(r[a].mean() - r[b].mean())
