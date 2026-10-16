# name: period34_motif_choice
# description: Among length-8 pairs where exactly one sequence is an exact repeat with period 3 or 4 (e.g. HHTHHTHH, HTTHHTTH) but not period 1 or 2, the rate of choosing that periodic sequence; observed below null means people spot and penalise short repeating motifs more than the model's motif generator predicts, above means less.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    def periodic(s):
        p1 = s.str[1:] == s.str[:-1]
        p2 = s.str[2:] == s.str[:-2]
        p3 = s.str[3:] == s.str[:-3]
        p4 = s.str[4:] == s.str[:-4]
        return ((p3 | p4) & ~p1 & ~p2).to_numpy()
    pa = periodic(a); pb = periodic(b)
    n = a.str.len().to_numpy()
    m = (n == 8) & (pa != pb)
    y = df["chose_left"].to_numpy()[m]
    ta = pa[m]
    if len(y) == 0:
        return 0.5
    return float(np.mean(np.where(ta, y, 1 - y)))
