# name: longest_run_avoidance_equal_switches
# description: Among pairs whose switch counts differ by at most 1 but whose longest runs differ by >=2 flips, the proportion of choices for the sequence with the SHORTER longest run; observed above null_mean means people penalise long streaks beyond what switch count and balance capture (model under-penalises streaks), below means the model over-penalises them.
def test_statistic(df):
    def feats(s):
        sw = sum(1 for x, y in zip(s, s[1:]) if x != y)
        best = cur = 1
        for x, y in zip(s, s[1:]):
            cur = cur + 1 if x == y else 1
            best = max(best, cur)
        return sw, best
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    m = {s: feats(s) for s in u}
    swa = df["sequence_a"].map(lambda s: m[s][0]).to_numpy()
    swb = df["sequence_b"].map(lambda s: m[s][0]).to_numpy()
    ra = df["sequence_a"].map(lambda s: m[s][1]).to_numpy()
    rb = df["sequence_b"].map(lambda s: m[s][1]).to_numpy()
    sel = (np.abs(swa - swb) <= 1) & (np.abs(ra - rb) >= 2)
    if sel.sum() == 0:
        return 0.5
    cl = df["chose_left"].to_numpy()[sel]
    c = np.where(ra[sel] < rb[sel], cl, 1 - cl)
    return float(np.mean(c))
