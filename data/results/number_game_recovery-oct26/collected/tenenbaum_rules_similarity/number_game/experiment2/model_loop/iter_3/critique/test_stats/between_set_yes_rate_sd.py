# name: between_set_yes_rate_sd
# description: Standard deviation across sets (with >=20 trials) of the per-set yes-rate on non-example targets; observed > null means people's overall willingness to generalize varies more across example sets (some sets read as tight rules, others as broad) than the model predicts, observed < null means the model's set-to-set breadth differences are exaggerated.
def test_statistic(df):
    exs = df["number_set"].map(lambda s: set(int(x) for x in s.split(","))).to_numpy()
    t = df["target"].astype(int).to_numpy()
    isex = np.array([tt in e for tt, e in zip(t, exs)])
    sub = df[~isex]
    g = sub.groupby("number_set")["rating"].agg(["mean", "size"])
    g = g[g["size"] >= 20]
    if len(g) < 2:
        return 0.0
    return float(g["mean"].std())
