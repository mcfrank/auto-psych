# name: periodic_pattern_choice_rate
# description: Among pairs where exactly one sequence is a repetition of a block of length 2..len/2 (e.g. HHTTHHTT, HTTHTT; perfect alternation excluded), the proportion choosing the periodic sequence; observed below null means people penalise visible repeating patterns that the model does not (model over-produces choice of periodic sequences).
import re
def _alt(s):
    return sum(1 for x, y in zip(s, s[1:]) if x != y) / (len(s) - 1)
def _imb(s):
    return abs(s.count("H") - s.count("T")) / len(s)
def _maxrun(s):
    return max(len(m.group(0)) for m in re.finditer(r"H+|T+", s))
def _feat(df, f):
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    m = {s: f(s) for s in u}
    return df["sequence_a"].map(m).to_numpy(float), df["sequence_b"].map(m).to_numpy(float)

def _periodic(s):
    n = len(s)
    for p in range(2, n // 2 + 1):
        if n % p == 0 and s[:p] * (n // p) == s and len(set(s[:p])) > 1:
            return 1.0
    return 0.0
def test_statistic(df):
    pa, pb = _feat(df, _periodic)
    mask = pa != pb
    if mask.sum() == 0:
        return 0.5
    c = df["chose_left"].to_numpy()
    return float(np.mean((c == (pa > pb).astype(int))[mask]))
