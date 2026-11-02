# name: within_cell_disagreement
# description: Mean over (set, target) cells with 2+ responses of the unbiased within-cell variance k(m-k)/(m(m-1)) of the yes responses; observed above null means respondents disagree with each other on the same item more than one shared posterior predicts (individual differences the model lacks), below null means they agree more.
def test_statistic(df):
    g = df.assign(_y=df["rating"].astype(float)).groupby(["number_set", "target"])["_y"].agg(["sum", "size"])
    g = g[g["size"] >= 2]
    k = g["sum"].to_numpy()
    m = g["size"].to_numpy().astype(float)
    return float(np.mean(k * (m - k) / (m * (m - 1)))) if len(g) else 0.0
