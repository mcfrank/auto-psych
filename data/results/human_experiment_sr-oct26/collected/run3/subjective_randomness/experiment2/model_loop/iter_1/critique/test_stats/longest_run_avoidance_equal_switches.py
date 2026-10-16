# name: longest_run_avoidance_equal_switches
# description: Among trials whose two sequences have the same number of switches but different longest runs, the proportion choosing the sequence with the shorter longest run; observed > null means the model under-weights streak (long-run) avoidance beyond switch count.
import re
def _maxrun(s):
    return max(len(m.group(0)) for m in re.finditer(r"H+|T+", s))
def _sw(s):
    return sum(1 for x, y in zip(s, s[1:]) if x != y)
def test_statistic(df):
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    mr = {s: _maxrun(s) for s in u}; sw = {s: _sw(s) for s in u}
    ra = df["sequence_a"].map(mr).to_numpy(); rb = df["sequence_b"].map(mr).to_numpy()
    ka = df["sequence_a"].map(sw).to_numpy(); kb = df["sequence_b"].map(sw).to_numpy()
    m = (ka == kb) & (ra != rb)
    if m.sum() == 0:
        return 0.5
    y = df["chose_left"].to_numpy()[m]
    chose_shorter = np.where(ra[m] < rb[m], y, 1 - y)
    return float(chose_shorter.mean())
