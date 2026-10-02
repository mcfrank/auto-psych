# name: shorter_initial_run_choice
# description: Among pairs whose opening runs (streak at the start) differ in length but whose longest runs are equal, the proportion choosing the sequence with the shorter opening run; observed above null_mean means people penalise starting streaks more than the model (which only weights the final run) predicts.
def test_statistic(df):
    def first_run(s):
        n = 1
        while n < len(s) and s[n] == s[0]:
            n += 1
        return n
    def longest(s):
        best = cur = 1
        for x, y in zip(s, s[1:]):
            cur = cur + 1 if x == y else 1
            best = max(best, cur)
        return best
    a = df["sequence_a"]; b = df["sequence_b"]
    u = pd.unique(pd.concat([a, b]))
    fr = {s: first_run(s) for s in u}; lr = {s: longest(s) for s in u}
    fa = a.map(fr).to_numpy(); fb = b.map(fr).to_numpy()
    la = a.map(lr).to_numpy(); lb = b.map(lr).to_numpy()
    sel = (fa != fb) & (la == lb)
    if sel.sum() == 0:
        return 0.5
    c = df["chose_left"].to_numpy()
    chose_shorter = np.where(fa < fb, c, 1 - c)
    return float(chose_shorter[sel].mean())
