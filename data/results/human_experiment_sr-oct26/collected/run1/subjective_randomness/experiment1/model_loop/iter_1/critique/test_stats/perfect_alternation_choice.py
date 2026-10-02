# name: perfect_alternation_choice
# description: Among pairs where exactly one sequence is perfectly alternating (alternation rate 1) and the other is streaky (alternation rate <= 2/7, e.g. HTHTHTHT vs HHHHHHHT), the proportion choosing the perfect alternator; observed above the null means the model under-predicts the population-level preference for perfect alternation over streaks (its split of personal ideals makes it too indifferent), below means it over-predicts it.

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
    aa = _feat(df["sequence_a"], _alt); ab = _feat(df["sequence_b"], _alt)
    y = df["chose_left"].to_numpy(dtype=float)
    m1 = (aa == 1) & (ab <= 2/7 + 1e-9)
    m2 = (ab == 1) & (aa <= 2/7 + 1e-9)
    n = m1.sum() + m2.sum()
    if n == 0:
        return 0.5
    return float((y[m1].sum() + (1 - y[m2]).sum()) / n)
