# name: high_alt_pref_both_above_half
# description: Among pairs where both sequences have alternation rate >= 0.5 and the rates differ, the proportion choosing the higher-alternation sequence; observed below null means people dislike over-alternation more sharply than the model's quadratic ideal-point allows (above null: less).
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

def test_statistic(df):
    aa, ab = _feat(df, _alt)
    mask = (aa >= 0.5) & (ab >= 0.5) & (np.abs(aa - ab) > 1e-9)
    if mask.sum() == 0:
        return 0.5
    c = df["chose_left"].to_numpy()
    return float(np.mean((c == (aa > ab).astype(int))[mask]))
