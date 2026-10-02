# name: short_seq_more_alternating_choice
# description: Among pairs of length <= 5 whose alternation rates differ, the proportion choosing the sequence with more alternations; observed far from null_mean means the model's length-normalised features mis-scale how strongly alternation drives choices for short sequences (above: people favour alternation more than predicted).
def test_statistic(df):
    def alt(s):
        return sum(1 for x, y in zip(s, s[1:]) if x != y) / (len(s) - 1)
    a = df["sequence_a"]; b = df["sequence_b"]
    m = {s: alt(s) for s in pd.unique(pd.concat([a, b]))}
    aa = a.map(m).to_numpy(); bb = b.map(m).to_numpy()
    L = a.str.len().to_numpy()
    sel = (L <= 5) & (aa != bb)
    if sel.sum() == 0:
        return 0.5
    chose_more = np.where(aa > bb, df["chose_left"].to_numpy(), 1 - df["chose_left"].to_numpy())
    return float(chose_more[sel].mean())
