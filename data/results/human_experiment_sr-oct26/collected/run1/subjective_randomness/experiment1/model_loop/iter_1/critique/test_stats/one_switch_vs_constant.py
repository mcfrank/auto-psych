# name: one_switch_vs_constant
# description: Among pairs where one sequence is constant (0 alternations) and the other has exactly one switch, the proportion choosing the one-switch sequence; observed above the model's null means the model under-predicts how strongly people reject all-same sequences.

def _alt(s):
    return sum(x != y for x, y in zip(s, s[1:])) / (len(s) - 1)

def _imb(s):
    return abs(s.count("H") - len(s) / 2.0) / len(s)

def _maxrun(s):
    best = cur = 1
    for x, y in zip(s, s[1:]):
        cur = cur + 1 if x == y else 1
        best = max(best, cur)
    return best

def _feat(col, f):
    u = pd.unique(col)
    return col.map({s: f(s) for s in u}).to_numpy(dtype=float)

def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    na = _feat(a, lambda s: sum(x != y for x, y in zip(s, s[1:])))
    nb = _feat(b, lambda s: sum(x != y for x, y in zip(s, s[1:])))
    y = df["chose_left"].to_numpy(dtype=float)
    m1 = (na == 1) & (nb == 0); m2 = (na == 0) & (nb == 1)
    n = m1.sum() + m2.sum()
    if n == 0:
        return 0.5
    return float((y[m1].sum() + (1 - y[m2]).sum()) / n)
