# name: small_target_yes_after_high_sets
# description: Yes-rate on targets 1-10 for sets whose smallest example is >=30; observed > null means people generalize to small numbers far below the examples (via broad rules) more than the model predicts, observed < null means less.
def test_statistic(df):
    mn = df["number_set"].map(lambda s: min(int(x) for x in s.split(","))).to_numpy()
    t = df["target"].astype(int).to_numpy()
    m = (mn >= 30) & (t <= 10)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy()[m].mean())
