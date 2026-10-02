# name: short_seq_alt_agreement
# description: Among trials with sequence length <= 5 whose alternation rates differ, the proportion choosing the higher-alternation sequence minus the same proportion among length-8 trials; observed above the null means short sequences are judged by alternation more (or long ones less) than the model's length-independent sensitivity predicts, below means the reverse.
def test_statistic(df):
    def alt(s):
        return sum(1 for x, y in zip(s, s[1:]) if x != y) / (len(s) - 1)
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    f = {s: alt(s) for s in u}
    aa = df["sequence_a"].map(f).to_numpy(float)
    ab = df["sequence_b"].map(f).to_numpy(float)
    L = df["sequence_a"].str.len().to_numpy()
    y = df["chose_left"].to_numpy(float)
    d = aa - ab
    hi = np.where(d > 0, y, 1 - y)
    ms = (L <= 5) & (np.abs(d) > 1e-9)
    ml = (L == 8) & (np.abs(d) > 1e-9)
    if ms.sum() == 0 or ml.sum() == 0:
        return 0.0
    return float(hi[ms].mean() - hi[ml].mean())
