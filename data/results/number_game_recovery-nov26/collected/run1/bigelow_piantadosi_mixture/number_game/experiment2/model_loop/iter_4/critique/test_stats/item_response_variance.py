# name: item_response_variance
# description: Mean, over (set, target) items answered >=4 times, of the binomial variance p(1-p) of the item's yes-rate; observed below null_mean means people agree on items more (responses more deterministic/extreme) than the model's intermediate probabilities predict, above means people are less consistent than the model.
def test_statistic(df):
    g = df.groupby([df["number_set"].astype(str), df["target"].astype(int)])["rating"].agg(["mean", "size"])
    g = g[g["size"] >= 4]
    if len(g) == 0:
        return 0.0
    p = g["mean"].values.astype(float)
    return float(np.mean(p * (1 - p)))
