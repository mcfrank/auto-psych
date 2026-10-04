# name: symmetric_pattern_avoidance
# description: Among pairs where exactly one sequence is a palindrome or a complement-mirror (e.g. HHTT-like reversals) but neither is periodic, the rate of choosing the non-symmetric sequence; observed above null means people penalize visible symmetry that the model (only periodicity and switching) does not capture.
def test_statistic(df):
    a, b = df["sequence_a"], df["sequence_b"]
    tr = str.maketrans("HT", "TH")
    def sym(x):
        r = x[::-1]
        return float(x == r or x == r.translate(tr))
    def per(x):
        n = len(x)
        return float(any(all(x[i] == x[i + p] for i in range(n - p)) for p in range(1, n // 2 + 1)))
    def f(s, g):
        return s.map({x: g(x) for x in s.unique()})
    ya, yb = f(a, sym), f(b, sym)
    mask = (ya != yb) & (f(a, per) == 0) & (f(b, per) == 0)
    if mask.sum() == 0:
        return 0.5
    c = df["chose_left"][mask]
    return float(np.where((ya < yb)[mask], c, 1 - c).mean())
