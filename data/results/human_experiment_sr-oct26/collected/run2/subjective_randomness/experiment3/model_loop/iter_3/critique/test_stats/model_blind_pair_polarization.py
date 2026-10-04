# name: model_blind_pair_polarization
# description: Among unordered pairs whose two sequences tie on the model cues (equal tally span share, equal switch count, equal three-flip chunk variety, equal rule-built status), the n-weighted mean squared deviation from 0.5 of the rate of choosing the alphabetically-first sequence; observed above null means people consistently discriminate pairs the model treats as equivalent (a missing cue), below means less.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    seqs = pd.unique(pd.concat([a, b]))
    def span(s):
        t = np.cumsum([1 if c == "H" else -1 for c in s])
        return round((max(t.max(), 0) - min(t.min(), 0)) / len(s), 9)
    def trip(s):
        n = len(s) - 2
        return 0.0 if n < 1 else round(len({s[i:i+3] for i in range(n)}) / min(8, n), 9)
    def rule(s):
        n = len(s)
        if len(set(s)) < 2:
            return 0
        for p in range(2, n // 2 + 1):
            if all(s[i] == s[i+p] for i in range(n - p)):
                return 1
        return int(s == s.translate(str.maketrans("HT", "TH"))[::-1])
    sig = {s: (span(s), sum(x != y for x, y in zip(s, s[1:])), trip(s), rule(s)) for s in seqs}
    m = (a.map(sig) == b.map(sig)) & (a != b)
    if m.sum() == 0:
        return 0.0
    aa = a[m]; bb = b[m]
    first = aa < bb
    key = pd.Series(np.where(first, aa + "|" + bb, bb + "|" + aa), index=aa.index)
    y = pd.Series(np.where(first, df["chose_left"][m], 1 - df["chose_left"][m]), index=aa.index)
    g = y.groupby(key).agg(["mean", "size"])
    return float(np.average((g["mean"] - 0.5) ** 2, weights=g["size"]))
