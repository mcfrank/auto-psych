# name: over_alternation_switch_preference
# description: Among trials where both sequences have >= 5 switches and the switch counts differ, the proportion choosing the sequence with MORE switches; observed < null means the model over-predicts preference for further alternation in already highly alternating sequences (people penalise over-alternation more), > null the reverse.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    def switches(s):
        n = s.str.len().max()
        p = s.str.pad(n, side="right", fillchar=".")
        tot = np.zeros(len(s))
        for i in range(n - 1):
            c1 = p.str[i].to_numpy(); c2 = p.str[i + 1].to_numpy()
            tot += ((c1 != c2) & (c1 != ".") & (c2 != ".")).astype(float)
        return tot
    ka = switches(a); kb = switches(b)
    m = (ka >= 5) & (kb >= 5) & (ka != kb)
    y = df["chose_left"].to_numpy()
    chose_more = np.where(ka > kb, y, 1 - y)[m]
    return float(chose_more.mean())
