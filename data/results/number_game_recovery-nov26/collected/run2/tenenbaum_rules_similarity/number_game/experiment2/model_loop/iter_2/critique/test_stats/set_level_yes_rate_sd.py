# name: set_level_yes_rate_sd
# description: SD across example sets (with at least 10 non-example-target trials) of each set's yes rate on non-example targets; observed above the null means sets differ in how broadly people generalize more than the model captures (model too homogeneous across sets), below means the model exaggerates set-to-set differences.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    exm = sets.str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy()
    inset = (exm == df["target"].astype(int).to_numpy()[:, None]).any(axis=1)
    sub = df[~inset]
    g = sub.groupby(sets[~inset])["rating"].agg(["mean", "size"])
    g = g[g["size"] >= 10]
    if len(g) < 2:
        return 0.0
    return float(g["mean"].std())
