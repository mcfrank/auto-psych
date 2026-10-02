# name: short_sequence_higher_alternation_rate
# description: Among trials with sequences of length <= 7 whose alternation rates differ, the proportion choosing the sequence with the HIGHER alternation rate; observed above the null means people's preference for switching is stronger on short sequences than the model's length-independent ideal-rate term predicts, below means weaker.
def test_statistic(df):
    def alt(s):
        return sum(1 for x, y in zip(s, s[1:]) if x != y) / (len(s) - 1)
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    m = {s: alt(s) for s in u}
    aa = df["sequence_a"].map(m).to_numpy(dtype=float)
    ab = df["sequence_b"].map(m).to_numpy(dtype=float)
    L = df["sequence_a"].str.len().to_numpy()
    sel = (L <= 7) & (aa != ab)
    if sel.sum() == 0:
        return 0.5
    c = df["chose_left"].to_numpy()
    chose = np.where(aa > ab, c, 1 - c)
    return float(chose[sel].mean())
