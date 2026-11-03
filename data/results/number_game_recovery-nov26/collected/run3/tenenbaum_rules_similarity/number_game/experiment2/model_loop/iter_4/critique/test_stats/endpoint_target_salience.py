# name: endpoint_target_salience
# description: Over sets not containing the target, yes rate on the line endpoints (targets 1 and 100) minus yes rate on near-endpoint targets (2-5 and 96-99); observed > null means people give the endpoints a salience bonus beyond what the model's concepts assign them, observed < null means they avoid the endpoints more.
def test_statistic(df):
    sets = df["number_set"].astype(str).values
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    isex = np.array([str(x) in s.split(",") for x, s in zip(t, sets)])
    end = ((t == 1) | (t == 100)) & ~isex
    near = (((t >= 2) & (t <= 5)) | ((t >= 96) & (t <= 99))) & ~isex
    if end.sum() == 0 or near.sum() == 0:
        return 0.0
    return float(r[end].mean() - r[near].mean())
