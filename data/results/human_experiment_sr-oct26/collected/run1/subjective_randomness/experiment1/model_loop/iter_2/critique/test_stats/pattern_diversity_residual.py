# name: pattern_diversity_residual
# description: Coefficient on (distinct 3-gram count left minus right, divided by length-2) in a linear-probability regression of chose_left on [1, alt diff, imbalance diff, 3-gram diversity diff] among length>=5 trials; observed above the null means people prefer sequences with more varied local patterns (less periodic/repetitive) beyond alternation and balance, which the model under-predicts.
def test_statistic(df):
    def feats(s):
        alt = sum(1 for x, y in zip(s, s[1:]) if x != y) / (len(s) - 1)
        imb = abs(s.count("H") - s.count("T")) / len(s)
        div = len({s[i:i + 3] for i in range(len(s) - 2)}) / (len(s) - 2) if len(s) >= 3 else 0.0
        return alt, imb, div
    d = df[df["sequence_a"].str.len() >= 5]
    u = pd.unique(pd.concat([d["sequence_a"], d["sequence_b"]]))
    f = {s: feats(s) for s in u}
    A = np.array([f[s] for s in d["sequence_a"]], dtype=float)
    B = np.array([f[s] for s in d["sequence_b"]], dtype=float)
    X = np.column_stack([np.ones(len(d)), A[:, 0] - B[:, 0], B[:, 1] - A[:, 1], A[:, 2] - B[:, 2]])
    y = d["chose_left"].to_numpy(dtype=float)
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(coef[3])
