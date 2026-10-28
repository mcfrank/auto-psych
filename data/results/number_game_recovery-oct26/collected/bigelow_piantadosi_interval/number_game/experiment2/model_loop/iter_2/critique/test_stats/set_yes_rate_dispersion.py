# name: set_yes_rate_dispersion
# description: SD across example sets (with >= 10 non-example trials) of the yes rate to non-example targets; observed above the null means people's generalization breadth differs between sets more than the model produces (model too uniform across sets), below means the model exaggerates set-to-set breadth differences.
def test_statistic(df):
    ex = df["number_set"].str.split(",")
    tgt = df["target"].astype(int).astype(str)
    is_ex = pd.Series([t in e for t, e in zip(tgt, ex)], index=df.index)
    d = df[~is_ex]
    g = d.groupby("number_set")["rating"].agg(["mean", "size"])
    g = g[g["size"] >= 10]
    return float(g["mean"].std()) if len(g) > 1 else 0.0
