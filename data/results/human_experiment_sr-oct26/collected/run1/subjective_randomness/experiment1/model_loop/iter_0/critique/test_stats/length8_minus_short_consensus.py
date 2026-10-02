# name: length8_minus_short_consensus
# description: Mean |pair-level choice rate - 0.5| for length-8 pairs minus that for shorter pairs (lengths 2-7); observed > null means people discriminate long sequences more (short ones less) than the model's single length-invariant beta allows, observed < null the reverse.
def test_statistic(df):
    a = df["sequence_a"].to_numpy()
    b = df["sequence_b"].to_numpy()
    y = df["chose_left"].to_numpy()
    first = np.where(a < b, a, b)
    second = np.where(a < b, b, a)
    chose_first = np.where(a < b, y, 1 - y)
    key = (pd.Series(first) + "|" + pd.Series(second)).to_numpy()
    rates = pd.Series(chose_first).groupby(key).mean()
    lens = pd.Series(df["sequence_a"].str.len().to_numpy()).groupby(key).first()
    dev = np.abs(rates - 0.5)
    long_ = dev[lens >= 8]
    short = dev[lens < 8]
    if len(long_) == 0 or len(short) == 0:
        return 0.0
    return float(long_.mean() - short.mean())
