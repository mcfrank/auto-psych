# name: item_rate_overdispersion
# description: Across distinct unordered pairs, the variance of the rate at which the alphabetically-first sequence is chosen (weighted by n); observed above null means items are more extreme/polarised than the model predicts (it misses stimulus features driving strong consensus), below null means it is too confident.
def test_statistic(df):
    a = df["sequence_a"].to_numpy(); b = df["sequence_b"].to_numpy()
    first = np.where(a < b, a, b); second = np.where(a < b, b, a)
    chose_first = np.where(a < b, df["chose_left"].to_numpy(), 1 - df["chose_left"].to_numpy())
    g = pd.DataFrame({"k": pd.Series(first) + "|" + pd.Series(second), "y": chose_first})
    s = g.groupby("k")["y"].agg(["mean", "size"])
    w = s["size"].to_numpy(); m = s["mean"].to_numpy()
    mu = np.sum(w * m) / w.sum()
    return float(np.sum(w * (m - mu) ** 2) / w.sum())
