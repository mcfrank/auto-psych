# name: short_pairs_more_switches
# description: Among pairs of length 2-3 with differing switch counts, the rate of choosing the sequence with more switches; observed above null means people prefer alternation in very short sequences more than the model's length-scaled evidence predicts, below means less.
def _switches(s):
    n = s.str.len().max()
    k = np.zeros(len(s))
    for i in range(1, n):
        x = s.str[i - 1].to_numpy()
        z = s.str[i].to_numpy()
        valid = s.str.len().to_numpy() > i
        k += valid & (x != z)
    return k
def test_statistic(df):
    L = df["sequence_a"].str.len().to_numpy()
    sub = df[L <= 3]
    ka = _switches(sub["sequence_a"])
    kb = _switches(sub["sequence_b"])
    m = ka != kb
    y = sub["chose_left"].to_numpy()
    chose_more = np.where(ka > kb, y, 1 - y)
    if m.sum() == 0:
        return 0.5
    return float(chose_more[m].mean())
