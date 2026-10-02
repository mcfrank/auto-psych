# name: pair_consensus_spread
# description: Variance across distinct pairs (side-independent) of the proportion choosing the lexicographically-first sequence, over pairs of length >= 7; observed above null_mean means people's pair-level preferences are more extreme/decisive than the model predicts (model too noisy or misordered), below means less extreme.
def test_statistic(df):
    a = df["sequence_a"].to_numpy()
    b = df["sequence_b"].to_numpy()
    keep = df["sequence_a"].str.len().to_numpy() >= 7
    first = np.where(a < b, a, b)
    key = first + "|" + np.where(a < b, b, a)
    chose_first = np.where(a < b, df["chose_left"].to_numpy(), 1 - df["chose_left"].to_numpy())
    r = pd.Series(chose_first[keep]).groupby(key[keep]).mean()
    if len(r) < 2:
        return 0.0
    return float(r.var())
