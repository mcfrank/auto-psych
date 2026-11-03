# name: endpoint_target_yes_deficit
# description: Yes rate for the endpoint targets 1 and 100 minus the overall yes rate; observed below null means people reject the extreme numbers 1 and 100 more than the model predicts, above means the model under-endorses them.
def test_statistic(df):
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    m = (t == 1) | (t == 100)
    if not m.any():
        return 0.0
    return float(y[m].mean() - y.mean())
