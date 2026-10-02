# name: pair_consensus_fraction
# description: Fraction of distinct unordered stimulus pairs on which at least 85% of responses pick the same sequence (side-invariant); observed above null means human consensus on some pairs is stronger than the model's lapse-diluted predictions allow (model under-produces near-unanimous pairs).
def test_statistic(df):
    a = df["sequence_a"].to_numpy(); b = df["sequence_b"].to_numpy()
    first = np.where(a < b, a, b); second = np.where(a < b, b, a)
    chose_first = np.where(a < b, df["chose_left"].to_numpy(), 1 - df["chose_left"].to_numpy())
    g = pd.Series(chose_first.astype(float)).groupby([first, second]).mean()
    return float(np.mean(np.maximum(g, 1 - g) >= 0.85))
