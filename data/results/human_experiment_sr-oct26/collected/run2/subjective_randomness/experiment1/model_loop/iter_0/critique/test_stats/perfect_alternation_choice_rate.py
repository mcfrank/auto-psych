# name: perfect_alternation_choice_rate
# description: Rate of choosing a perfectly alternating sequence (HTHT.../THTH...) when it is paired with a non-perfectly-alternating one; observed > null means the model under-predicts acceptance of the extreme over-alternating pattern, observed < null means it over-predicts it.
def test_statistic(df):
    def perf(s):
        return s.map({q: float(all(q[i] != q[i - 1] for i in range(1, len(q)))) for q in s.unique()}).to_numpy()
    pa = perf(df["sequence_a"]); pb = perf(df["sequence_b"])
    m = pa != pb
    if m.sum() == 0:
        return 0.5
    y = df["chose_left"].to_numpy()
    c = np.where(pa > pb, y, 1 - y)
    return float(c[m].mean())
