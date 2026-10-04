# name: shorter_run_rate_matched_alternations
# description: Among trials whose two sequences have equal alternation counts but different longest-run lengths, the rate of choosing the sequence with the shorter longest run; observed > null means the model under-weights long streaks as a cue to non-randomness.
def test_statistic(df):
    def feats(s):
        u = s.unique()
        alt = {}; run = {}
        for q in u:
            alt[q] = sum(q[i] != q[i - 1] for i in range(1, len(q)))
            best = cur = 1
            for i in range(1, len(q)):
                cur = cur + 1 if q[i] == q[i - 1] else 1
                best = max(best, cur)
            run[q] = best
        return s.map(alt).to_numpy(), s.map(run).to_numpy()
    aa, ra = feats(df["sequence_a"]); ab, rb = feats(df["sequence_b"])
    m = (aa == ab) & (ra != rb)
    if m.sum() == 0:
        return 0.5
    y = df["chose_left"].to_numpy()
    c = np.where(ra < rb, y, 1 - y)
    return float(c[m].mean())
