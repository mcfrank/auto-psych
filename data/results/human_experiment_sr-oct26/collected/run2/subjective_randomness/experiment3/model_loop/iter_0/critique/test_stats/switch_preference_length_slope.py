# name: switch_preference_length_slope
# description: Slope (per flip of length) of the rate of choosing the more-switching sequence among pairs with unequal switch rates; observed above null means the preference for switching grows with length faster than the model predicts (below null: slower or reverses).
def test_statistic(df):
    seqs = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    sw = {s: sum(x != y for x, y in zip(s, s[1:])) / (len(s) - 1) for s in seqs}
    ra = df["sequence_a"].map(sw).to_numpy(); rb = df["sequence_b"].map(sw).to_numpy()
    L = df["sequence_a"].str.len().to_numpy().astype(float)
    m = ra != rb
    y = np.where(ra > rb, df["chose_left"].to_numpy(), 1 - df["chose_left"].to_numpy())[m].astype(float)
    x = L[m]
    if x.size < 3 or np.var(x) == 0:
        return 0.0
    return float(np.cov(x, y, bias=True)[0, 1] / np.var(x))
