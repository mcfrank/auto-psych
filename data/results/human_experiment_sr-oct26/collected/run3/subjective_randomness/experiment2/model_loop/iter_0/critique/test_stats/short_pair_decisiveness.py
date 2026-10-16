# name: short_pair_decisiveness
# description: Mean over distinct unordered pairs of length <= 3 of |pair choice rate - 0.5|; observed above null_mean means people are more decisive on short pairs than the model predicts (below: less decisive).
def test_statistic(df):
    a = df["sequence_a"].values; b = df["sequence_b"].values
    m = df["sequence_a"].str.len().values <= 3
    if m.sum() == 0:
        return 0.0
    first = np.where(a < b, a, b); second = np.where(a < b, b, a)
    cf = np.where(a < b, df["chose_left"].values, 1 - df["chose_left"].values)
    key = (pd.Series(first) + "|" + pd.Series(second)).values
    r = pd.Series(cf[m]).groupby(key[m]).mean()
    return float(np.mean(np.abs(r.values - 0.5)))
