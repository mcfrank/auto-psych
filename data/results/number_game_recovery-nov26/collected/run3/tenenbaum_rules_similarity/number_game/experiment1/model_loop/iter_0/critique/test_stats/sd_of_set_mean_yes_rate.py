# name: sd_of_set_mean_yes_rate
# description: Standard deviation across example sets (with >=10 responses) of each set's mean yes rate; observed below the model mean means the model differentiates sets (broad vs narrow concepts) more sharply than people do.
def test_statistic(df):
    codes, uniq = pd.factorize(df["number_set"].astype(str))
    E = np.full((len(uniq), 4), np.nan)
    for i, s in enumerate(uniq):
        v = [int(x) for x in s.split(",")][:4]
        E[i, :len(v)] = v
    X = E[codes]  # (rows, 4) examples, NaN padded
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(X), axis=1)
    g = pd.DataFrame({"c": codes, "r": r}).groupby("c")["r"].agg(["mean", "size"])
    g = g[g["size"] >= 10]
    return float(g["mean"].std()) if len(g) > 1 else 0.0
