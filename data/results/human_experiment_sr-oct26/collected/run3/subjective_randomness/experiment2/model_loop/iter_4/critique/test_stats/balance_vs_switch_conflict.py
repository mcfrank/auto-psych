# name: balance_vs_switch_conflict
# description: Among conflict pairs where the sequence with more switches is also less balanced (|#H - #T| larger), the proportion choosing the more-switching sequence; observed < null means people weigh H/T imbalance more heavily against randomness than the model does, observed > null that they weigh it less (switches dominate).
def test_statistic(df):
    def arr(s):
        n = s.str.len().max()
        return np.array([list(x.ljust(n, "_")) for x in s])
    A = arr(df["sequence_a"]); B = arr(df["sequence_b"])
    def k(X):
        return ((X[:, 1:] != X[:, :-1]) & (X[:, 1:] != "_")).sum(1)
    def imb(X):
        return np.abs((X == "H").sum(1) - (X == "T").sum(1))
    ka, kb, ia, ib = k(A), k(B), imb(A), imb(B)
    m = ((ka > kb) & (ia > ib)) | ((kb > ka) & (ib > ia))
    y = df["chose_left"].to_numpy()
    return float(np.where(ka > kb, y, 1 - y)[m].mean())
