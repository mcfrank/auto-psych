# name: perfect_balance_bonus
# description: Among pairs with alternation rates within 1/7 where one sequence has exactly equal H and T counts and the other has imbalance 0.25 or less (but nonzero), the proportion choosing the perfectly balanced one; observed above the null means exact balance carries a bonus beyond the model's linear imbalance penalty , below the null means the model over-predicts the preference for exact balance.
def test_statistic(df):
    def feats(s):
        alt = sum(1 for x, y in zip(s, s[1:]) if x != y) / (len(s) - 1)
        imb = abs(s.count("H") - s.count("T")) / len(s)
        return alt, imb
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    f = {s: feats(s) for s in u}
    A = np.array([f[s] for s in df["sequence_a"]], dtype=float)
    B = np.array([f[s] for s in df["sequence_b"]], dtype=float)
    y = df["chose_left"].to_numpy(float)
    close = np.abs(A[:, 0] - B[:, 0]) <= 1 / 7 + 1e-9
    a_bal = (A[:, 1] < 1e-9) & (B[:, 1] > 1e-9) & (B[:, 1] <= 0.25 + 1e-9)
    b_bal = (B[:, 1] < 1e-9) & (A[:, 1] > 1e-9) & (A[:, 1] <= 0.25 + 1e-9)
    m = close & (a_bal | b_bal)
    if m.sum() == 0:
        return 0.5
    chose_bal = np.where(a_bal, y, 1 - y)
    return float(chose_bal[m].mean())
