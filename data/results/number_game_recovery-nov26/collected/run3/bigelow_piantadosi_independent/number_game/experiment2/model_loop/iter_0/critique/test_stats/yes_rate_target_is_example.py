# name: yes_rate_target_is_example
# description: Yes rate on trials where the target is itself one of the shown examples; observed below null means the model (near-zero lapse, all-or-none commitment) over-predicts endorsing the examples themselves (people say no to ~17% of them).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float)
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    mind = ex.sub(t, axis=0).abs().min(axis=1)
    m = (mind == 0).values
    return float(r[m].mean()) if m.sum() > 0 else 0.0
