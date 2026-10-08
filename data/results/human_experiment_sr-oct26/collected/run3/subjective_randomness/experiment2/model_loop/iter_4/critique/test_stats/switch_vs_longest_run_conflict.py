# name: switch_vs_longest_run_conflict
# description: Among conflict pairs where the sequence with more switches also contains the longer longest run, the proportion choosing the more-switching sequence; observed < null means people weigh a long streak more heavily against randomness than the model's switch/motif evidence does, observed > null that they weigh the switch count more.
def test_statistic(df):
    def arr(s):
        n = s.str.len().max()
        return np.array([list(x.ljust(n, "_")) for x in s])
    def stats(X):
        sw = (X[:, 1:] != X[:, :-1]) & (X[:, 1:] != "_")
        k = sw.sum(1)
        n = X.shape[1]
        run = np.ones(X.shape[0]); cur = np.ones(X.shape[0])
        for j in range(1, n):
            cont = (X[:, j] == X[:, j - 1]) & (X[:, j] != "_")
            cur = np.where(cont, cur + 1, 1)
            run = np.maximum(run, cur)
        return k, run
    ka, la = stats(arr(df["sequence_a"]))
    kb, lb = stats(arr(df["sequence_b"]))
    m = ((ka > kb) & (la > lb)) | ((kb > ka) & (lb > la))
    y = df["chose_left"].to_numpy()
    return float(np.where(ka > kb, y, 1 - y)[m].mean())
