# name: end_streak_pref_matched_alt
# description: Among pairs with alternation rates within 0.15 where exactly one sequence starts or ends with a run of 3+ identical flips, the proportion choosing the sequence WITHOUT an edge streak; observed above null means streaks at the sequence edges are penalised beyond the model's alternation/balance terms.
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

def _edge(s):
    return float(bool(re.match(r"^(HHH|TTT)", s)) or bool(re.search(r"(HHH|TTT)$", s)))
def test_statistic(df):
    aa, ab = _feat(df, _alt); ea, eb = _feat(df, _edge)
    mask = (np.abs(aa - ab) <= 0.15 + 1e-9) & (ea != eb)
    if mask.sum() == 0:
        return 0.5
    c = df["chose_left"].to_numpy()
    return float(np.mean((c == (ea < eb).astype(int))[mask]))
