# name: participant_sd_maxrun_pref
# description: Across participants, the SD of each person's rate of choosing the sequence with the shorter longest run (pairs differing in longest run); observed above null means individuals differ in streak aversion more than the model's balance/alternation/lapse heterogeneity produces.
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
    ma, mb = _feat(df, _maxrun)
    mask = ma != mb
    c = df["chose_left"].to_numpy()
    agree = (c == (ma < mb).astype(int)).astype(float)
    s = pd.Series(agree[mask]).groupby(df["participant_id"].to_numpy()[mask]).mean()
    return float(s.std(ddof=1)) if len(s) > 1 else 0.0
