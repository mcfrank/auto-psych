# name: example_target_yes_rate
# description: Yes-rate on trials whose target is one of the shown examples; observed > model means people endorse already-shown examples more reliably than the model (whose guessing dilutes these near-certain items) predicts, observed < model means they endorse them less.
def test_statistic(df):
    s = "," + df["number_set"].astype(str).str.replace(" ", "") + ","
    t = "," + df["target"].astype(int).astype(str) + ","
    mask = np.array([b in a for a, b in zip(s.values, t.values)])
    if mask.sum() == 0:
        return 0.0
    return float(df["rating"].values[mask].mean())
