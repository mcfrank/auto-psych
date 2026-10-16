# name: first_run_shorter_preference_equal_switches
# description: Among pairs with equal switch counts whose opening runs (leading identical flips) differ in length, the proportion choosing the sequence with the shorter opening run; observed > null means people penalise a streak at the start more than the model (which ignores where runs sit) predicts, < null that they penalise it less.
def test_statistic(df):
    def arr(s):
        n = s.str.len().max()
        return np.array([list(x.ljust(n, "_")) for x in s])
    A = arr(df["sequence_a"]); B = arr(df["sequence_b"])
    def switches(X):
        return ((X[:, 1:] != X[:, :-1]) & (X[:, 1:] != "_")).sum(1)
    def first_run(X):
        same = (X == X[:, :1])
        return np.cumprod(same, axis=1).sum(1)
    ka, kb = switches(A), switches(B)
    ra, rb = first_run(A), first_run(B)
    m = (ka == kb) & (ra != rb)
    y = df["chose_left"].to_numpy()
    return float(np.where(ra < rb, y, 1 - y)[m].mean())
