# name: high_alternation_preference
# description: Among trials where both sequences have alternation proportion >= 0.5 and differ in alternation count, the rate of choosing the more-alternating one; observed > null means humans prefer heavier alternation than the model's fitted alternation prototype allows, observed < null means the model over-rewards alternation at the high end.
def test_statistic(df):
    def pa(s):
        return s.map({q: sum(q[i] != q[i - 1] for i in range(1, len(q))) / max(len(q) - 1, 1) for q in s.unique()}).to_numpy()
    a = pa(df["sequence_a"]); b = pa(df["sequence_b"])
    m = (a >= 0.5) & (b >= 0.5) & (a != b)
    if m.sum() == 0:
        return 0.5
    y = df["chose_left"].to_numpy()
    c = np.where(a > b, y, 1 - y)
    return float(c[m].mean())
