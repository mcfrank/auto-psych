# name: maxrun_residual_after_alt_balance
# description: Coefficient on (longest run right minus longest run left) in a linear-probability regression of chose_left on [1, alt_left-alt_right, imbalance_right-imbalance_left, maxrun_right-maxrun_left] over all trials; observed above the null means people penalise long runs beyond alternation rate and H/T balance, which the model under-predicts.
def test_statistic(df):
    import re
    def feats(s):
        alt = sum(1 for x, y in zip(s, s[1:]) if x != y) / (len(s) - 1)
        imb = abs(s.count("H") - s.count("T")) / len(s)
        mr = max(len(m.group(0)) for m in re.finditer(r"H+|T+", s))
        return alt, imb, mr
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    f = {s: feats(s) for s in u}
    A = np.array([f[s] for s in df["sequence_a"]], dtype=float)
    B = np.array([f[s] for s in df["sequence_b"]], dtype=float)
    X = np.column_stack([np.ones(len(df)), A[:, 0] - B[:, 0], B[:, 1] - A[:, 1], B[:, 2] - A[:, 2]])
    y = df["chose_left"].to_numpy(dtype=float)
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(coef[3])
