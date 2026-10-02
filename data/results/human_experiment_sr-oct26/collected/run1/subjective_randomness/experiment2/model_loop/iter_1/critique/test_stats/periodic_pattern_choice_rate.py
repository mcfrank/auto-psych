# name: periodic_pattern_choice_rate
# description: Among trials where exactly one sequence is built by repeating a non-constant chunk of length 2-4 (e.g. HTTHTTHT, HHTTHHTT, HTHHTHHT; strict alternation excluded), the proportion choosing the periodic sequence; observed below the null means people detect and penalise repeated chunks beyond what the model's alternation/streak terms predict.
def test_statistic(df):
    def periodic(s):
        n = len(s)
        if len(set(s)) == 1 or all(x != y for x, y in zip(s, s[1:])):
            return False
        for p in (2, 3, 4):
            if n >= 2 * p and all(s[i] == s[i - p] for i in range(p, n)):
                return True
        return False
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    m = {s: periodic(s) for s in u}
    a = df["sequence_a"].map(m).astype(bool).to_numpy()
    b = df["sequence_b"].map(m).astype(bool).to_numpy()
    sel = a ^ b
    if sel.sum() == 0:
        return 0.5
    c = df["chose_left"].to_numpy()
    chose = np.where(a, c, 1 - c)
    return float(chose[sel].mean())
