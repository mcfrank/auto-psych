# name: lone_example_above_minus_below
# description: On one-example sets, yes rate for targets larger than the example minus yes rate for targets smaller than it; observed above null means the model under-predicts upward (vs downward) generalization from a lone example.
def test_statistic(df):
    n = df["number_set"].astype(str).str.count(",") + 1
    sub = df[n == 1]
    x = pd.to_numeric(sub["number_set"], errors="coerce").to_numpy()
    t = sub["target"].astype(int).to_numpy()
    r = sub["rating"].to_numpy().astype(float)
    a, b = t > x, t < x
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
