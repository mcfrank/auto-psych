# name: cell_response_extremity
# description: Mean over (set, target) cells of |cell yes rate - 0.5|; observed below null means the model's predictions are over-confident (too close to 0/1, lacking a lapse/noise component), above null means people are more deterministic than the model.
def test_statistic(df):
    key = df["number_set"].astype(str) + "|" + df["target"].astype(str)
    cm = df["rating"].astype(float).groupby(key).mean()
    return float(np.abs(cm.to_numpy() - 0.5).mean())
