# name: balance_rate_matched_alternations
# description: Among trials whose sequences have equal alternation counts but different global |#H-#T|, the rate of choosing the globally more balanced sequence; observed > null means the model under-weights global H/T balance (its multiscale averaging dilutes it), observed < null means it over-weights it.
def test_statistic(df):
    def f(s):
        u = s.unique()
        alt = {q: sum(q[i] != q[i - 1] for i in range(1, len(q))) for q in u}
        imb = {q: abs(2 * q.count("H") - len(q)) for q in u}
        return s.map(alt).to_numpy(), s.map(imb).to_numpy()
    aa, ia = f(df["sequence_a"]); ab, ib = f(df["sequence_b"])
    m = (aa == ab) & (ia != ib)
    if m.sum() == 0:
        return 0.5
    y = df["chose_left"].to_numpy()
    c = np.where(ia < ib, y, 1 - y)
    return float(c[m].mean())
