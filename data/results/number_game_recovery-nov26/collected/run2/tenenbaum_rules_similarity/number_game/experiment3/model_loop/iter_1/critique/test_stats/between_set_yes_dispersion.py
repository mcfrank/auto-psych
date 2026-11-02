# name: between_set_yes_dispersion
# description: Standard deviation across sets (those with at least 40 non-example-target trials) of the per-set yes-rate; observed above null means the model under-produces how differently people treat different sets (too homogeneous), below means it exaggerates set differences.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).astype(float).to_numpy()
    t = df["target"].astype(float).to_numpy()
    inset = np.where(~np.isnan(ex), ex == t[:, None], False).any(axis=1)
    d = df.loc[~inset, ["number_set", "rating"]]
    g = d.groupby("number_set")["rating"].agg(["mean", "size"])
    g = g[g["size"] >= 40]
    if len(g) < 2:
        return 0.0
    return float(g["mean"].astype(float).std())
