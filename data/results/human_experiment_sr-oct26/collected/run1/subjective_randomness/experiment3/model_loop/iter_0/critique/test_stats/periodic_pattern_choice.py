# name: periodic_pattern_choice
# description: Among pairs where exactly one sequence repeats a block of period 3 or 4 (e.g. HHTHHT, HTTTHTTT; not strict alternation), the proportion choosing the periodic one; observed below null_mean means people see repeating patterns as non-random more than the model (no periodicity term) predicts.
def test_statistic(df):
    def periodic(s):
        n = len(s)
        if n < 6:
            return False
        if all(s[i] == s[i + 2] for i in range(n - 2)):
            return False
        for p in (3, 4):
            if n >= 2 * p - 1 and all(s[i] == s[i + p] for i in range(n - p)):
                return True
        return False
    a = df["sequence_a"]; b = df["sequence_b"]
    m = {s: periodic(s) for s in pd.unique(pd.concat([a, b]))}
    pa = a.map(m).to_numpy(bool); pb = b.map(m).to_numpy(bool)
    sel = pa != pb
    if sel.sum() == 0:
        return 0.5
    c = df["chose_left"].to_numpy()
    chose_per = np.where(pa, c, 1 - c)
    return float(chose_per[sel].mean())
