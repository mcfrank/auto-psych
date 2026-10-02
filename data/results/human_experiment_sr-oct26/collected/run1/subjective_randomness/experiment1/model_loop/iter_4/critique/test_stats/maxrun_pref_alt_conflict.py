# name: maxrun_pref_alt_conflict
# description: Among pairs whose longest runs differ and where the shorter-longest-run sequence does NOT have the higher alternation rate (streak and alternation cues conflict or tie), the proportion choosing the shorter-longest-run sequence; observed above null means people weigh streak length beyond alternation/balance (model under-produces streak aversion), below null the reverse.
import re
def _alt(s):
    return sum(1 for x, y in zip(s, s[1:]) if x != y) / (len(s) - 1)
def _maxrun(s):
    return max(len(m.group(0)) for m in re.finditer(r"H+|T+", s))
def _feat(df, f):
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    m = {s: f(s) for s in u}
    return df["sequence_a"].map(m).to_numpy(float), df["sequence_b"].map(m).to_numpy(float)
def test_statistic(df):
    aa, ab = _feat(df, _alt); ma, mb = _feat(df, _maxrun)
    ls = ma < mb
    alt_s = np.where(ls, aa, ab); alt_l = np.where(ls, ab, aa)
    mask = (ma != mb) & (alt_s <= alt_l + 1e-9)
    if mask.sum() == 0:
        return 0.5
    c = df["chose_left"].to_numpy()
    return float(np.mean((c == ls.astype(int))[mask]))
