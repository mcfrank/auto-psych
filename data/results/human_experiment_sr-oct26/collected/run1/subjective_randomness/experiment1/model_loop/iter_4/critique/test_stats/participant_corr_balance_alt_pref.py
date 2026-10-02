# name: participant_corr_balance_alt_pref
# description: Across participants, the Pearson correlation between their rate of choosing the more balanced sequence (pairs differing in imbalance) and their rate of choosing the sequence with alternation closer to 0.6 (pairs differing in that distance); observed above null means a shared "engagement/sensitivity" trait links the two preferences that the model's independent person-level parameters do not produce.
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
    aa, ab = _feat(df, _alt); ia, ib = _feat(df, _imb)
    c = df["chose_left"].to_numpy(); pid = df["participant_id"].to_numpy()
    mb = np.abs(ia - ib) > 1e-9
    b_agree = (c == (ia < ib).astype(int)).astype(float)
    da, db = np.abs(aa - 0.6), np.abs(ab - 0.6)
    ma = np.abs(da - db) > 1e-9
    a_agree = (c == (da < db).astype(int)).astype(float)
    sb = pd.Series(b_agree[mb]).groupby(pid[mb]).mean()
    sa = pd.Series(a_agree[ma]).groupby(pid[ma]).mean()
    j = pd.concat([sb, sa], axis=1, join="inner").dropna()
    if len(j) < 3 or j.iloc[:, 0].std() == 0 or j.iloc[:, 1].std() == 0:
        return 0.0
    return float(np.corrcoef(j.iloc[:, 0], j.iloc[:, 1])[0, 1])
