# name: switch_clustering_preference_equal_switches
# description: Among trials whose sequences have the same number of switches but different (switch-after-switch minus repeat-after-switch) counts, the proportion choosing the sequence whose switches come more in consecutive runs; observed < null means people dislike back-to-back switches more than the model's shared second-order term predicts, > null less.
def test_statistic(df):
    def stats(s):
        n = s.str.len().to_numpy()
        L = int(n.max())
        arr = np.array([list(x.ljust(L, "?")) for x in s])
        valid = (arr[:, 1:] != "?") & (arr[:, :-1] != "?")
        tr = ((arr[:, 1:] != arr[:, :-1]) & valid).astype(int)
        v2 = valid[:, 1:] & valid[:, :-1]
        ss = ((tr[:, 1:] == 1) & (tr[:, :-1] == 1) & v2).sum(1)
        sr = ((tr[:, :-1] == 1) & (tr[:, 1:] == 0) & v2).sum(1)
        return tr.sum(1), ss - sr
    ka, aa = stats(df["sequence_a"])
    kb, ab = stats(df["sequence_b"])
    m = (ka == kb) & (aa != ab)
    y = df["chose_left"].to_numpy()
    return float(np.where(aa > ab, y, 1 - y)[m].mean())
