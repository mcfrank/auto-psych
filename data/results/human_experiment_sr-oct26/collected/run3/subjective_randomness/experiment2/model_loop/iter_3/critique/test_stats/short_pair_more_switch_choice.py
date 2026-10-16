# name: short_pair_more_switch_choice
# description: Among trials of length 2 or 3 whose sequences differ in switch count, the proportion choosing the sequence with more switches; observed > null means the model's length scaling under-predicts how strongly people favour alternation in very short sequences, < null over-predicts it.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    n = a.str.len().to_numpy()
    def k(s):
        L = int(s.str.len().max())
        arr = np.array([list(x.ljust(L, "?")) for x in s])
        valid = (arr[:, 1:] != "?") & (arr[:, :-1] != "?")
        return ((arr[:, 1:] != arr[:, :-1]) & valid).sum(1)
    ka, kb = k(a), k(b)
    m = (n <= 3) & (ka != kb)
    y = df["chose_left"].to_numpy()
    return float(np.where(ka > kb, y, 1 - y)[m].mean())
