# name: short_sequence_switchier_choice
# description: Among trials of length <= 5 whose sequences differ in switch count, the proportion choosing the sequence with MORE switches; observed above null_mean means on short sequences people favour alternation more than the model (whose ideal switching rate is shared across lengths) predicts, below means less.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    L = a.str.len()
    def sw(s):
        u = pd.unique(s)
        m = {x: sum(1 for i in range(len(x) - 1) if x[i] != x[i + 1]) for x in u}
        return s.map(m).to_numpy()
    sa, sb = sw(a), sw(b)
    mask = (L.to_numpy() <= 5) & (sa != sb)
    if not mask.any():
        return 0.5
    c = df["chose_left"].to_numpy()
    chose_more = np.where(sa > sb, c, 1 - c)
    return float(chose_more[mask].mean())
