# name: agreement_sd_structurally_neutral
# description: SD across participants of the rate of agreeing with the pair majority, restricted to pairs where both sequences have equal three-flip chunk variety and neither/both are rule-built (so pattern sensitivity cannot act); observed above null means person heterogeneity in consistency persists outside pattern structure (model under-produces it there), below means it over-produces it.
def test_statistic(df):
    seqs = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    def trip(s):
        n = len(s) - 2
        return 0.0 if n < 1 else len({s[i:i+3] for i in range(n)}) / min(8, n)
    def rule(s):
        n = len(s)
        if len(set(s)) < 2:
            return 0
        for p in range(2, n // 2 + 1):
            if all(s[i] == s[i+p] for i in range(n - p)):
                return 1
        return int(s == s.translate(str.maketrans("HT", "TH"))[::-1])
    tr = {s: trip(s) for s in seqs}; ru = {s: rule(s) for s in seqs}
    m = ((df["sequence_a"].map(tr) - df["sequence_b"].map(tr)).abs() < 1e-9) & \
        (df["sequence_a"].map(ru) == df["sequence_b"].map(ru))
    d = df[m.to_numpy()]
    a = d["sequence_a"]; b = d["sequence_b"]
    first = a < b
    key = pd.Series(np.where(first, a + "|" + b, b + "|" + a), index=d.index)
    y = pd.Series(np.where(first, d["chose_left"], 1 - d["chose_left"]), index=d.index)
    rate = y.groupby(key).transform("mean")
    agree = ((y == 1) == (rate >= 0.5)).astype(float)
    per = agree.groupby(d["participant_id"]).mean()
    return float(per.std()) if len(per) > 1 else 0.0
