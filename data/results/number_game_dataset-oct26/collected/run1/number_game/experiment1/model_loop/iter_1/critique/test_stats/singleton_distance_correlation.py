# name: singleton_distance_correlation
# description: For 1-example sets only, Pearson correlation between yes response and |target - example|; strongly negative means magnitude-driven generalization, so an observed value above (closer to 0 than) the null means the model over-weights numeric proximity for single examples, below means it under-weights it.
def test_statistic(df):
    n = df["number_set"].str.count(",") + 1
    sub = df[n == 1]
    if len(sub) < 3:
        return 0.0
    d = np.abs(sub["target"].astype(float).to_numpy() - pd.to_numeric(sub["number_set"]).astype(float).to_numpy())
    y = sub["rating"].astype(float).to_numpy()
    if d.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(d, y)[0, 1])
